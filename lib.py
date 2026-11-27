import numpy as np
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
from collections import deque
import os
import bm25s
from tqdm.auto import tqdm




class SALBER_BM25:

  def __init__(
      self,
      df_a: pd.DataFrame,
      df_b: pd.DataFrame,
      gold_standard: pd.DataFrame,
      id_col_a: str,
      id_col_b: str,
      text_col_a: str = 'text',
      text_col_b: str = 'text',
      epsilon_P: float = 0.05,
      epsilon_R: float = 0.05,
      delta: float = 0.05,
      m: int = 200,
      m_v: int = 200,
      k_neighbors: int = 10,
      K: int = 10,
      ds: str = '',
      eta: float = 0.01,
      index_dir: str = './bm25_indices/',
      lower_sim: float = 0.0,
  ):
        # ==========================================
        # 1. Tokenization & BM25s Indexing
        # ==========================================
        self.id_col_a = id_col_a
        self.id_col_b = id_col_b
        self.k_neighbors = k_neighbors
        self.lower_sim = lower_sim

        corpus_b = df_b[text_col_b].astype(str).tolist()
        queries_a = df_a[text_col_a].astype(str).tolist()

        corpus_b_tokens = bm25s.tokenize(corpus_b, stopwords='en', stemmer=None)
        queries_a_tokens = bm25s.tokenize(queries_a, stopwords='en', stemmer=None)

        save_path = os.path.join(index_dir, f'bm25_index_{ds}')
        if os.path.exists(save_path):
            print(f'Loading existing BM25 index from {save_path} ...')
            retriever = bm25s.BM25.load(save_path, load_corpus=False)
        else:
            print('Building BM25 index over df_b...')
            retriever = bm25s.BM25()
            retriever.index(corpus_b_tokens)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
                print(f'Saving BM25 index to {save_path} ...')
                retriever.save(save_path)

        print('Querying BM25 index for top-k candidates...')
        neighbor_indices, raw_scores = retriever.retrieve(
            queries_a_tokens, k=k_neighbors
        )

        print('Formulating candidate pairs dataframe...')
        if id_col_a in df_a.columns:
            a_ids = df_a[id_col_a].astype(str).values
        else:
            a_ids = df_a.index.astype(str).values

        if id_col_b in df_b.columns:
            b_ids = df_b[id_col_b].astype(str).values
        else:
            b_ids = df_b.index.astype(str).values

        A_col = np.repeat(a_ids, k_neighbors)
        B_col = b_ids[neighbor_indices.ravel()]
        score_col = raw_scores.ravel().astype(np.float32)

        candidates = pd.DataFrame(
            {id_col_a: A_col, id_col_b: B_col, 'raw_score': score_col}
        )

        # Normalize BM25 scores to [0, 1] for stable IPW binning and threshold search
        s_min, s_max = candidates['raw_score'].min(), candidates['raw_score'].max()
        if s_max > s_min:
            candidates['score'] = (candidates['raw_score'] - s_min) / (s_max - s_min)
        else:
            candidates['score'] = 1.0

        print(f'BM25 Blocking complete. Generated {len(candidates)} candidate pairs.')
        candidates = candidates[candidates['score'] > self.lower_sim].reset_index(drop=True)
        print(f'Using {len(candidates)} candidate pairs with score > {self.lower_sim}.')

        self.C = candidates.copy()
        self.N_pool = len(self.C)

        # ==========================================
        # 2. Oracle Matching & Constraints
        # ==========================================
        gs_a = gold_standard[self.id_col_a].astype(str)
        gs_b = gold_standard[self.id_col_b].astype(str)
        self.oracle_set = set(zip(gs_a, gs_b))

        # --- DIAGNOSTIC ONLY (Never used for dynamic_goal to prevent label leakage) ---
        C_tuples = set(zip(self.C[self.id_col_a], self.C[self.id_col_b]))
        self.recalled_matches_diagnostic = len(self.oracle_set.intersection(C_tuples))
        print(f'\n[DIAGNOSTIC] Total True matches in Gold Standard: {len(self.oracle_set)}')
        print(f'[DIAGNOSTIC] True matches successfully retrieved by BM25: {self.recalled_matches_diagnostic}\n')
        print(f'[TARGETS] Target Conditional Recall: >= {1.0 - epsilon_R:.2f}')

        self.eps_P = epsilon_P
        self.epsilon_op = epsilon_R
        self.eps_R = epsilon_R
        self.delta = delta
        self.eta = eta
        self.m = m
        self.m_v = m_v
        self.V_samples = []

        # ==========================================
        # 3. Similarity Buckets & Defensive Importance Sampling
        # ==========================================
        self.C['bucket'] = pd.cut(self.C['score'], bins=K, labels=False)
        unique_buckets = self.C['bucket'].unique()
        n_buckets = len(unique_buckets)
        bucket_sizes = self.C['bucket'].value_counts().to_dict()

        # 50/50 Defensive Mixture: guarantees raw IPW weights < 2.0 with zero clipping
        self.bucket_probs = {
            b: 0.5 * (1.0 / n_buckets) + 0.5 * (bucket_sizes[b] / self.N_pool)
            for b in unique_buckets
        }

        self.C['q_i'] = self.C['bucket'].map(lambda b: self.bucket_probs[b] / bucket_sizes[b])
        self.C['norm_w_i'] = 1.0 / (self.N_pool * self.C['q_i'])

        self.C['in_D'] = False
        self.C['in_V'] = False
        self.C['true_label'] = np.nan

        self.tau = 0.5
        self.t = 0

  def _query_oracle(self, idx_list):
        pairs_A = self.C.loc[idx_list, self.id_col_a].values
        pairs_B = self.C.loc[idx_list, self.id_col_b].values
        labels = [1.0 if pair in self.oracle_set else 0.0 for pair in zip(pairs_A, pairs_B)]
        self.C.loc[idx_list, 'true_label'] = labels

  def _retrain_hypothesis(self) -> float:
        D = self.C[self.C['in_D']]
        if len(D) == 0:
            return self.tau

        scores = D['score'].values
        labels = D['true_label'].values
        base_weights = D['norm_w_i'].values

        # Hájek Normalization retained exclusively to stabilize the asymmetric threshold optimizer
        base_weights = base_weights * (len(D) / np.sum(base_weights))

        pos_mask_weights = (labels == 1.0)
        neg_mask_weights = (labels == 0.0)

        pos_weight_sum = np.sum(base_weights[pos_mask_weights])
        neg_weight_sum = np.sum(base_weights[neg_mask_weights])

        if pos_weight_sum > 0:
            dynamic_ratio = 8.0 * (neg_weight_sum / pos_weight_sum)
        else:
            dynamic_ratio = 1.0

        candidate_taus = np.unique(scores)
        preds = scores[None, :] >= candidate_taus[:, None]
        fn_mask = (labels == 1.0) & ~preds
        fp_mask = (labels == 0.0) & preds

        fn_cost = np.sum(fn_mask * base_weights, axis=1) * dynamic_ratio
        fp_cost = np.sum(fp_mask * base_weights, axis=1)
        total_weighted_error = fn_cost + fp_cost

        best_idx = np.argmin(total_weighted_error)
        return float(candidate_taus[best_idx])

  def fit(self, seed_size: int = 200):
        time_start = time.time()
        p_seed = self.C['q_i'].values
        p_seed = p_seed / p_seed.sum()

        seed_idx = np.random.choice(self.C.index, size=seed_size, replace=False, p=p_seed)
        self.C.loc[seed_idx, 'in_D'] = True
        self._query_oracle(seed_idx)
        self.tau = self._retrain_hypothesis()
        print(f'Initialization complete. Starting tau_1 = {self.tau:.4f}')

        self.trajectory_log = []
        trigger_count = 0
        previous_tau = self.tau
        est_global_recall = 0.0
        ess = 0.0

        while True:
            self.t += 1

            # --- PHASE 1: HEURISTIC OPTIMIZATION (THE TRIGGER) ---
            margin_mask = (
                (np.abs(self.C['score'] - self.tau) <= self.eta)
                & (~self.C['in_D'])
                & (~self.C.index.isin(self.V_samples))
            )
            margin_candidates = self.C[margin_mask].index

            if len(margin_candidates) > 0:
                draw_b = min(self.m, len(margin_candidates))
                d_draws = np.random.choice(margin_candidates, size=draw_b, replace=False)
                self.C.loc[d_draws, 'in_D'] = True

                unlabeled_d = [idx for idx in d_draws if pd.isna(self.C.loc[idx, 'true_label'])]
                if unlabeled_d:
                    self._query_oracle(unlabeled_d)

            self.tau = self._retrain_hypothesis()

            print(
                f'[TRAIN t={self.t}] Threshold (tau): {self.tau:.4f} | '
                f'New margin pairs drawn: {min(self.m, len(margin_candidates)) if len(margin_candidates) > 0 else 0}'
            )

            if abs(self.tau - previous_tau) < 1e-2:
                trigger_count += 1
            else:
                trigger_count = 0

            previous_tau = self.tau

            if trigger_count >= 3:
                print(f'\n[TRIGGER] Threshold stabilized at {self.tau:.4f}. Initiating Strict Certification...')

                # Offline global ground-truth recall over C (diagnostic only)
                if hasattr(self, 'oracle_set') and self.oracle_set is not None:
                    is_true_match = np.array(
                        [(a, b) in self.oracle_set for a, b in zip(self.C[self.id_col_a], self.C[self.id_col_b])],
                        dtype=bool,
                    )
                    total_matches_C = int(is_true_match.sum())
                    actual_global_recall = (
                        float((is_true_match & (self.C['score'].values >= self.tau)).sum() / total_matches_C)
                        if total_matches_C > 0 else float('nan')
                    )
                else:
                    actual_global_recall = float('nan')

                # --- PHASE 2: STATISTICAL CERTIFICATION (THE BOUND) ---
                certified = False
                cert_samples = []
                self.t_cert = 0

                # Precompute Phase 2 invariants once (in_D and tau are frozen)
                mask_D = self.C['in_D'].values
                N_avail = self.N_pool - int(mask_D.sum())
                p_dist_cert = self.C['q_i'].values.copy()
                p_dist_cert[mask_D] = 0.0

                if N_avail == 0 or p_dist_cert.sum() == 0:
                    print('Candidate pool exhausted for certification.')
                    certified = True
                else:
                    p_dist_cert = p_dist_cert / p_dist_cert.sum()
                    dynamic_weights = np.zeros_like(p_dist_cert)
                    valid_mask = p_dist_cert > 0
                    dynamic_weights[valid_mask] = 1.0 / (N_avail * p_dist_cert[valid_mask])
                    c_cert = float(dynamic_weights[valid_mask].max())

                    D_set = self.C[mask_D]
                    FN_D = float(np.sum((D_set['true_label'].values == 1.0) & (D_set['score'].values < self.tau))) if len(D_set) > 0 else 0.0
                    G_D = float((D_set['true_label'] == 1.0).sum()) if len(D_set) > 0 else 0.0

                    while True:
                        self.t_cert += 1

                        v_draws = np.random.choice(self.C.index, size=self.m_v, replace=True, p=p_dist_cert)
                        cert_samples.extend(v_draws)
                        self.V_samples.extend(v_draws)
                        self.C.loc[v_draws, 'in_V'] = True

                        unlabeled_v = [idx for idx in v_draws if pd.isna(self.C.loc[idx, 'true_label'])]
                        if unlabeled_v:
                            self._query_oracle(unlabeled_v)

                        V_cert = self.C.loc[cert_samples]
                        N_v = len(V_cert)

                        preds = (V_cert['score'].values >= self.tau).astype(float)
                        labels = V_cert['true_label'].values
                        weights = dynamic_weights[cert_samples]
                        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2) if np.sum(weights) > 0 else 0.0

                        # Time-uniform failure budget (1-indexed validation step t_cert >= 1)
                        delta_t = (6.0 * (self.delta / 2.0)) / ((np.pi ** 2) * (self.t_cert ** 2))
                        log_term = np.log(2.0 / delta_t)

                        # 1. Upper Bound: False Negatives (Global Scale)
                        loss_R_array = weights * ((labels == 1.0) & (preds == 0.0)).astype(float)
                        E_hat_err = np.mean(loss_R_array)
                        S_err = np.var(loss_R_array, ddof=1)
                        var_penalty_err = np.sqrt((2.0 * S_err * log_term) / N_v)
                        scale_penalty_err = (7.0 * c_cert * log_term) / (3.0 * (N_v - 1))
                        U_R_residual = E_hat_err + var_penalty_err + scale_penalty_err
                        U_global = (FN_D + (N_avail * U_R_residual)) / self.N_pool

                        # 2. Lower Bound: True Matches (Global Scale)
                        match_array = weights * (labels == 1.0).astype(float)
                        E_hat_match = np.mean(match_array)
                        S_match = np.var(match_array, ddof=1)
                        var_penalty_match = np.sqrt((2.0 * S_match * log_term) / N_v)
                        scale_penalty_match = (7.0 * c_cert * log_term) / (3.0 * (N_v - 1))
                        L_M_residual = max(0.0, E_hat_match - (var_penalty_match + scale_penalty_match))
                        L_global = (G_D + (N_avail * L_M_residual)) / self.N_pool

                        # 3. Stopping Condition
                        dynamic_goal = self.epsilon_op * L_global

                        batch_matches = int((self.C.loc[v_draws, 'true_label'] == 1.0).sum())
                        cum_unique_matches = int((self.C.loc[list(set(cert_samples)), 'true_label'] == 1.0).sum())
                        print(
                            f'[VALID t={self.t_cert}] | c={c_cert:.4f} | Batch matches={batch_matches} | '
                            f'Cumulative unique matches={cum_unique_matches} | U_R_adjusted={U_global:.6f} | '
                            f'epsilon_t (Goal)={dynamic_goal:.6f}'
                        )

                        est_fn_global = FN_D + N_avail * E_hat_err
                        est_matches_global = G_D + N_avail * E_hat_match
                        est_global_recall = 1.0 - (est_fn_global / max(est_matches_global, 1e-12))
                        cert_global_recall = max(0.0, 1.0 - (U_global / L_global)) if L_global > 0 else 0.0
                        print(
                            f'  Est Recall (Global): {est_global_recall:.4f} | '
                            f'Cert Recall (Global): {cert_global_recall:.4f} | '
                            f'Global Recall (tau^*): {actual_global_recall:.4f} | '
                            f'Est Matches (Global): {est_matches_global:.4f}'
                        )

                        if U_global <= dynamic_goal:
                            print(f'\n[CONVERGENCE] Statistically Certified! U_global {U_global:.6f} <= Goal {dynamic_goal:.6f}')
                            certified = True
                            break

                        if self.t_cert >= 500:
                            break

                if certified:
                    break
                else:
                    print('[OVERFITTING DETECTED] Bound failed to converge. Resuming optimization.')
                    self.C.loc[cert_samples, 'in_D'] = True
                    trigger_count = 0

            if self.t >= 1000:
                print('\n[TIMEOUT] Reached maximum allowed optimization iterations.')
                break

        trajectory_df = pd.DataFrame(self.trajectory_log)

        unseen = self.C[self.C['true_label'].isna()]
        if len(unseen) > 0:
            y_true_unseen = np.array([
                1.0 if p in self.oracle_set else 0.0
                for p in zip(unseen[self.id_col_a], unseen[self.id_col_b])
            ])
            y_pred_unseen = (unseen['score'].values >= self.tau).astype(float)

            TP = np.sum((y_pred_unseen == 1.0) & (y_true_unseen == 1.0))
            FP = np.sum((y_pred_unseen == 1.0) & (y_true_unseen == 0.0))
            FN = np.sum((y_pred_unseen == 0.0) & (y_true_unseen == 1.0))

            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            print(f'\n--- Final Evaluation on Unseen Candidates ({len(unseen):,}) ---')
            print(f'True Positives: {TP} | False Positives: {FP} | False Negatives: {FN}')
            print(f'Precision: {prec:.4f} | Recall: {rec:.4f}')
        else:
            prec, rec = 0.0, 0.0
            print('\n--- Final Evaluation on Unseen Candidates (0) ---')

        print('\n--- Final Sample Counts ---')
        print(f"Total Training Samples (|M|): {self.C['in_D'].sum()}")
        print(f'Total Certification Samples (|V|): {len(set(self.V_samples))}')

        total_time = time.time() - time_start
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {
            'total_time': total_time,
            'estimated_recall': est_global_recall,
            'actual_recall': rec,
            'actual_precision': prec,
            'f1': f1,
            'ess': ess,
            'queries': int((self.C['in_V'] | self.C['in_D']).sum()),
            'positive_yield': int(self.C['true_label'].sum()),
        }

        return self.tau, self.C, metrics, trajectory_df





class SALBER:

    def _encode_to_memmap(
      self,
      texts_series,
      filepath: str,
      chunk_size: int = 50_000,
      batch_size: int = 256,
    ) -> np.ndarray:
        """Encodes a pandas Series in chunks directly into a standard .npy file on disk."""
        n_rows = len(texts_series)
        dim = self.model.get_embedding_dimension()
        tmp_path = f"{filepath}.tmp.npy"

        # Use FP16 on GPU for ~2x throughput and half the VRAM
        if hasattr(self.model, "device") and self.model.device.type == "cuda":
            self.model.half()

        # Pre-allocate a valid .npy file on disk without consuming system RAM
        mmap_emb = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(n_rows, dim)
        )

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            # Converting only the active slice to a list keeps Python memory negligible
            batch_texts = texts_series.iloc[start:end].tolist()
            emb_chunk = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            mmap_emb[start:end] = np.asarray(emb_chunk, dtype=np.float32)

        mmap_emb.flush()
        del mmap_emb

        # Atomic rename ensures partial/crashed runs never leave a corrupted cache file
        os.replace(tmp_path, filepath)
        return np.load(filepath, mmap_mode="r")



    def __init__(self,
                 df_a: pd.DataFrame,
                 df_b: pd.DataFrame,
                 gold_standard: pd.DataFrame,
                 id_col_a: str,
                 id_col_b: str,
                 model_name: str = 'all-MiniLM-L6-v2', #'BAAI/bge-small-en-v1.5'
                 epsilon_P: float = 0.05,
                 epsilon_R: float = 0.05,
                 delta: float = 0.05,
                 m: int = 200,
                 m_v: int = 200,
                 k_neighbors:int=10,
                 K: int = 10,
                 ds:str="",
                 eta:float=0.01,
                 emb_dir:str="./data/",
                 lower_sim:float=0.4):

        self.model = SentenceTransformer(model_name)
        print(f"Working on {ds}")
        # ==========================================
        # 1. Embedding Generation & HNSW Blocking
        # ==========================================
        os.makedirs(emb_dir, exist_ok=True)
        chunk_size = 100_000
        batch_size = 256
        M = 32
        path_emb_a = os.path.join(emb_dir, f"embeddings_{ds}_a.npy")
        path_emb_b = os.path.join(emb_dir, f"embeddings_{ds}_b.npy")
        path_index = os.path.join(emb_dir, f"hnsw_{ds}.index")

        # 1. Encode or memory-map Dataset A
        if os.path.exists(path_emb_a):
          print("Memory-mapping existing embeddings A ...")
          emb_a = np.load(path_emb_a, mmap_mode="r")
        else:
          print("Ecoding and streaming dataset A embeddings to disk ...")
          emb_a = self._encode_to_memmap(df_a["text"], path_emb_a, chunk_size, batch_size)

        # 2. Encode or memory-map Dataset B
        if os.path.exists(path_emb_b):
          print("Memory-mapping existing embeddings B ...")
          emb_b = np.load(path_emb_b, mmap_mode="r")
        else:
          print("Encoding and streaming dataset B embeddings to disk ...")
          emb_b = self._encode_to_memmap(df_b["text"], path_emb_b, chunk_size, batch_size)

        # 3. Build (and cache) or load the HNSW graph index
        if os.path.exists(path_index):
          print("Loading existing HNSW graph index ...")
          index = faiss.read_index(path_index)
        else:
          print("Building HNSW graph index ...")
          d = emb_b.shape[1]
          index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
          #index.hnsw.efConstruction = 200

          # Add vectors in chunks so emb_b is never duplicated in system RAM
          with tqdm(total=len(emb_b), desc="Indexing HNSW", unit="vec") as pbar:
            for start in range(0, len(emb_b), chunk_size):
               end = min(start + chunk_size, len(emb_b))
               chunk_b = np.ascontiguousarray(emb_b[start:end], dtype=np.float32)
               index.add(chunk_b)
               pbar.update(end - start)

          tmp_index_path = f"{path_index}.tmp"
          faiss.write_index(index, tmp_index_path)
          os.replace(tmp_index_path, path_index)

        # 4. Query the HNSW index in chunks
        print("Querying HNSW index for top-k candidates ...")
        self.k_neighbors = k_neighbors
        index.hnsw.efSearch = 128

        n_queries = len(emb_a)
        similarities = np.empty((n_queries, k_neighbors), dtype=np.float32)
        neighbor_indices = np.empty((n_queries, k_neighbors), dtype=np.int64)

        with tqdm(total=n_queries, desc="Querying HNSW", unit="query") as pbar:
          for start in range(0, n_queries, chunk_size):
             end = min(start + chunk_size, n_queries)
             chunk_a = np.ascontiguousarray(emb_a[start:end], dtype=np.float32)
             sim_chunk, idx_chunk = index.search(chunk_a, k_neighbors)
             similarities[start:end] = sim_chunk
             neighbor_indices[start:end] = idx_chunk
             pbar.update(end - start)


        print("Formulating candidate pairs dataframe...")

        if id_col_a in df_a.columns:
            a_ids = df_a[id_col_a].astype(str).values
        else:
            a_ids = df_a.index.astype(str).values

        if id_col_b in df_b.columns:
            b_ids = df_b[id_col_b].astype(str).values
        else:
            b_ids = df_b.index.astype(str).values

        A_col = np.repeat(a_ids, k_neighbors)
        B_col = b_ids[neighbor_indices.ravel()]
        score_col = similarities.ravel()

        candidates = pd.DataFrame({
            id_col_a: A_col,
            id_col_b: B_col,
            'score': score_col
        })
        self.lower_sim = lower_sim
        print(f"HNSW Blocking complete. Generated {len(candidates)} candidate pairs.")
        candidates = candidates[candidates['score'] > self.lower_sim].reset_index(drop=True)
        print(f"Using {len(candidates)} candidate pairs.")

        self.C = candidates.copy()
        self.N_pool = len(self.C)
        self.id_col_a = id_col_a
        self.id_col_b = id_col_b

        # ==========================================
        # 2. Oracle Matching & Constraints
        # ==========================================
        gs_a = gold_standard[self.id_col_a].astype(str)
        gs_b = gold_standard[self.id_col_b].astype(str)
        self.oracle_set = set(zip(gs_a, gs_b))

        # --- DIAGNOSTIC ONLY (Never used for dynamic_goal to prevent label leakage) ---
        C_tuples = set(zip(self.C[self.id_col_a], self.C[self.id_col_b]))
        self.recalled_matches_diagnostic = len(self.oracle_set.intersection(C_tuples))
        print(f"\n[DIAGNOSTIC] Total True matches in Gold Standard: {len(self.oracle_set)}")
        print(f"[DIAGNOSTIC] True matches successfully retrieved by HNSW: {self.recalled_matches_diagnostic}\n")
        print(f"[TARGETS] Target Conditional Recall: >= {1.0 - epsilon_R:.2f}")

        self.eps_P = epsilon_P
        self.epsilon_op = epsilon_R
        self.eps_R = epsilon_R
        self.delta = delta
        self.eta = eta
        self.m = m
        self.m_v = m_v
        self.V_samples = []

        # ==========================================
        # 3. Similarity Buckets & Defensive Importance Sampling
        # ==========================================
        self.C['bucket'] = pd.cut(self.C['score'], bins=K, labels=False)
        unique_buckets = self.C['bucket'].unique()
        n_buckets = len(unique_buckets)
        bucket_sizes = self.C['bucket'].value_counts().to_dict()

        # 50/50 Defensive Mixture: guarantees raw IPW weights < 2.0 with zero clipping
        self.bucket_probs = {
            b: 0.5 * (1.0 / n_buckets) + 0.5 * (bucket_sizes[b] / self.N_pool)
            for b in unique_buckets
        }

        self.C['q_i'] = self.C['bucket'].map(lambda b: self.bucket_probs[b] / bucket_sizes[b])
        self.C['norm_w_i'] = 1.0 / (self.N_pool * self.C['q_i'])

        self.C['in_D'] = False
        self.C['in_V'] = False
        self.C['true_label'] = np.nan

        self.tau = 0.5
        self.t = 0

    def _query_oracle(self, idx_list):
        pairs_A = self.C.loc[idx_list, self.id_col_a].values
        pairs_B = self.C.loc[idx_list, self.id_col_b].values
        labels = [1.0 if pair in self.oracle_set else 0.0 for pair in zip(pairs_A, pairs_B)]
        self.C.loc[idx_list, 'true_label'] = labels

    def _retrain_hypothesis(self):
        D = self.C[self.C['in_D']]
        if len(D) == 0:
            return self.tau

        scores = D['score'].values
        labels = D['true_label'].values
        base_weights = D['norm_w_i'].values
        # Hájek Normalization is retained here exclusively to stabilize the asymmetric threshold optimizer
        base_weights = base_weights * (len(D) / np.sum(base_weights))
        pos_mask_weights = (labels == 1.0)
        neg_mask_weights = (labels == 0.0)

        pos_weight_sum = np.sum(base_weights[pos_mask_weights])
        neg_weight_sum = np.sum(base_weights[neg_mask_weights])

        if pos_weight_sum > 0:
            dynamic_ratio = 8 * (neg_weight_sum / pos_weight_sum)
        else:
            dynamic_ratio = 1.0

        candidate_taus = np.unique(scores)
        preds = scores[None, :] >= candidate_taus[:, None]
        fn_mask = (labels == 1.0) & ~preds
        fp_mask = (labels == 0.0) & preds

        fn_cost = np.sum(fn_mask * base_weights, axis=1) * dynamic_ratio
        fp_cost = np.sum(fp_mask * base_weights, axis=1)
        total_weighted_error = fn_cost + fp_cost

        best_idx = np.argmin(total_weighted_error)
        best_tau = candidate_taus[best_idx]

        return float(best_tau)



    def fit(self, seed_size: int = 200):
        time_start = time.time()
        p_seed = self.C['q_i'].values
        p_seed = p_seed / p_seed.sum()

        seed_idx = np.random.choice(self.C.index, size=seed_size, replace=False, p=p_seed)

        self.C.loc[seed_idx, 'in_D'] = True
        self._query_oracle(seed_idx)
        self.tau = self._retrain_hypothesis()
        print(f"Initialization complete. Starting tau_1 = {self.tau:.4f}")

        self.trajectory_log = []
        trigger_count = 0
        previous_tau = self.tau

        while True:
            self.t += 1

            # --- PHASE 1: HEURISTIC OPTIMIZATION (THE TRIGGER) ---
            # Active Margin Sampling
            margin_mask = (np.abs(self.C['score'] - self.tau) <= self.eta) & (~self.C['in_D']) & (~self.C.index.isin(self.V_samples))
            margin_candidates = self.C[margin_mask].index

            if len(margin_candidates) > 0:
                draw_b = min(self.m, len(margin_candidates))
                d_draws = np.random.choice(margin_candidates, size=draw_b, replace=False)
                self.C.loc[d_draws, 'in_D'] = True

                unlabeled_d = [idx for idx in d_draws if pd.isna(self.C.loc[idx, 'true_label'])]
                if unlabeled_d:
                    self._query_oracle(unlabeled_d)

            # Update threshold
            self.tau = self._retrain_hypothesis()

            # Log the training trajectory
            print(f"[TRAIN t={self.t}] Threshold (tau): {self.tau:.4f} | New margin pairs drawn: {min(self.m, len(margin_candidates)) if len(margin_candidates) > 0 else 0}")

            # Check stability (Trigger condition)
            if abs(self.tau - previous_tau) < 1e-2:
                trigger_count += 1
            else:
                trigger_count = 0

            previous_tau = self.tau

            if trigger_count >= 3:
                num_train = self.C['in_D'].sum()
                print(f"\n[TRIGGER] Threshold stabilized at {self.tau:.4f}. Initiating Strict Certification...")
                # Exact counts in the labeled active training set D (evaluated at frozen self.tau)
                # Offline global ground-truth recall over C (constant across Phase 2 since self.tau is frozen)
                if hasattr(self, "oracle_set") and self.oracle_set is not None:
                    is_true_match = np.array(
                        [(a, b) in self.oracle_set for a, b in zip(self.C[self.id_col_a], self.C[self.id_col_b])],
                        dtype=bool,
                    )
                    total_matches_C = int(is_true_match.sum())
                    actual_global_recall = (
                        float((is_true_match & (self.C["score"].values >= self.tau)).sum() / total_matches_C)
                        if total_matches_C > 0 else float("nan")
                    )
                else:
                    actual_global_recall = float("nan")
                # --- PHASE 2: STATISTICAL CERTIFICATION (THE BOUND) ---
                certified = False
                cert_samples = []
                self.t_cert = 0

                while True:
                    self.t_cert += 1
                    # Draw a fresh batch strictly outside of the training set D
                    p_dist = self.C['q_i'].values
                    # Mask out training samples from probability distribution
                    mask_D = self.C['in_D'].values
                    N_avail = self.N_pool - mask_D.sum()
                    p_dist_cert = p_dist.copy()
                    p_dist_cert[mask_D] = 0.0
                    if p_dist_cert.sum() == 0:
                         print("Candidate pool exhausted for certification.")
                         certified = True # Fallback if pool is exhausted
                         break
                    p_dist_cert = p_dist_cert / p_dist_cert.sum()
                    dynamic_weights = np.zeros_like(p_dist_cert)
                    valid_mask = p_dist_cert > 0
                    dynamic_weights[valid_mask] = 1.0 / (N_avail * p_dist_cert[valid_mask])
                    c_cert = float(dynamic_weights[valid_mask].max())

                    v_draws = np.random.choice(self.C.index, size=self.m_v, replace=True, p=p_dist_cert)
                    cert_samples.extend(v_draws)
                    self.V_samples.extend(v_draws) # Keep track for global isolation if it fails
                    self.C.loc[v_draws, 'in_V'] = True

                    unlabeled_v = [idx for idx in v_draws if pd.isna(self.C.loc[idx, 'true_label'])]
                    if unlabeled_v:
                        self._query_oracle(unlabeled_v)

                    # Evaluate on the accumulated certification set against frozen tau
                    V_cert = self.C.loc[cert_samples]
                    N_v = len(V_cert)

                    preds = (V_cert['score'].values >= self.tau).astype(float)
                    labels = V_cert['true_label'].values
                    weights = dynamic_weights[cert_samples]
                    ess = (np.sum(weights)**2) / np.sum(weights**2) if np.sum(weights) > 0 else 0.0

                    # Time-uniform failure budget (1-indexed validation step t_cert >= 1)
                    delta_t = (6.0 * (self.delta / 2.0)) / ((np.pi ** 2) * (self.t_cert ** 2))
                    log_term = np.log(2.0 / delta_t)
                    D_set = self.C[self.C['in_D']]
                    # ---------------------------------------------------------
                    # 1. THE UPPER BOUND: False Negatives (Global Scale)
                    # ---------------------------------------------------------
                    loss_R_array = weights * ((labels == 1.0) & (preds == 0.0)).astype(float)
                    E_hat_err = np.mean(loss_R_array)
                    S_err = np.var(loss_R_array, ddof=1)
                    var_penalty_err = np.sqrt((2.0 * S_err * log_term) / N_v)
                    scale_penalty_err = (7.0 * c_cert * log_term) / (3.0 * (N_v - 1))
                    U_R_residual = E_hat_err + var_penalty_err + scale_penalty_err
                    FN_D = np.sum((D_set['true_label'].values == 1.0) & (D_set['score'].values < self.tau)) if len(D_set) > 0 else 0.0
                    U_global = (FN_D + (N_avail * U_R_residual)) / self.N_pool
                    # ---------------------------------------------------------
                    # 2. THE LOWER BOUND: True Matches (Global Scale)
                    # ---------------------------------------------------------
                    match_array = weights * (labels == 1.0).astype(float)
                    E_hat_match = np.mean(match_array)
                    S_match = np.var(match_array, ddof=1)
                    var_penalty_match = np.sqrt((2.0 * S_match * log_term) / N_v)
                    scale_penalty_match = (7.0 * c_cert * log_term) / (3.0 * (N_v - 1))
                    L_M_residual = max(0.0, E_hat_match - (var_penalty_match + scale_penalty_match))
                    G_D = (D_set['true_label'] == 1.0).sum() if len(D_set) > 0 else 0.0
                    L_global = (G_D + (N_avail * L_M_residual)) / self.N_pool




                    # ---------------------------------------------------------
                    # 3. STOPPING CONDITION
                    # ---------------------------------------------------------
                    dynamic_goal = self.epsilon_op * L_global

                    batch_matches = int((self.C.loc[v_draws, 'true_label'] == 1.0).sum())
                    cum_unique_matches = int((self.C.loc[list(set(cert_samples)), 'true_label'] == 1.0).sum())
                    # Log the metrics for this certification iteration
                    print(f"[VALID t={self.t_cert}] | c={c_cert:.4f} | Batch matches={batch_matches} | Cumulative unique matches={cum_unique_matches} | U_R_adjusted={U_global:.6f} | epsilon_t (Goal)={dynamic_goal:.6f}")
                    # Inside each Phase 2 validation iteration t:
                    est_fn_global = FN_D + N_avail * E_hat_err
                    est_matches_global = G_D + N_avail * E_hat_match
                    est_global_recall = 1.0 - (est_fn_global / max(est_matches_global, 1e-12))
                    cert_global_recall = max(0.0, 1.0 - (U_global / L_global)) if L_global > 0 else 0.0
                    print(
                        f"  Est Recall (Global): {est_global_recall:.4f} | "
                        f"  Cert Recall (Global): {cert_global_recall:.4f} | "
                        f"  Global Recall (tau^*): {actual_global_recall:.4f} | "
                        f"  Est Matches (Global): {est_matches_global:.4f} | "
                    )


                    if U_global <= dynamic_goal:
                        print(f"\n[CONVERGENCE] Statistically Certified! U_global {U_global:.6f} <= Goal {dynamic_goal:.6f}")
                        certified = True
                        break

                    # If it hasn't certified after max iterations, it's an overfitting artifact
                    if self.t_cert >= 500: # Arbitrary max cert steps to prevent infinite stalling
                         break

                if certified:
                    break
                else:
                    print("[OVERFITTING DETECTED] Bound failed to converge. Resuming optimization.")
                    # Dump failed certification batch into training set D
                    self.C.loc[cert_samples, 'in_D'] = True
                    trigger_count = 0 # Reset trigger

            if self.t >= 1000:
                print("\n[TIMEOUT] Reached maximum allowed optimization iterations.")
                break

        trajectory_df = pd.DataFrame(self.trajectory_log)

        unseen = self.C[self.C['true_label'].isna()]

        if len(unseen) > 0:
            y_true_unseen = np.array([1.0 if p in self.oracle_set else 0.0 for p in zip(unseen[self.id_col_a], unseen[self.id_col_b])])
            y_pred_unseen = (unseen['score'].values >= self.tau).astype(float)

            TP = np.sum((y_pred_unseen == 1.0) & (y_true_unseen == 1.0))
            FP = np.sum((y_pred_unseen == 1.0) & (y_true_unseen == 0.0))
            FN = np.sum((y_pred_unseen == 0.0) & (y_true_unseen == 1.0))

            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            print(f"\n--- Final Evaluation on Unseen Candidates ({len(unseen):,}) ---")
            print(f"True Positives: {TP} | False Positives: {FP} | False Negatives: {FN}")
            print(f"Precision: {prec:.4f} | Recall: {rec:.4f}")
        else:
            prec, rec = 0.0, 0.0
            print("\n--- Final Evaluation on Unseen Candidates (0) ---")

        print(f"\n--- Final Sample Counts ---")
        print(f"Total Training Samples (|M|): {self.C['in_D'].sum()}")
        print(f"Total Certification Samples (|V|): {len(set(self.V_samples))}")

        time_end = time.time()
        total_time = time_end - time_start
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Note: ess calculation might need adjustment depending on scope
        metrics = {
          'total_time': total_time,
          'estimated_recall': 0.0, # Update this based on final cert evaluation if needed
          'actual_recall': rec,
          'actual_precision': prec,
          'f1': f1,
          'ess' : 0.0,
          'queries': len(self.C[self.C['in_V'] | self.C['in_D']]),
          'positive_yield': self.C['true_label'].sum(),
        }

        return self.tau, self.C, metrics, trajectory_df






def plot_roc(C_df, df_full, ds):
    """
    Computes and plots the True Global ROC vs. SALBER's IPW-Adjusted ROC.
    """
    # 1. True Global ROC (from the fully-labeled offline reference)
    fpr_true, tpr_true, _ = roc_curve(df_full['label'], df_full['score'])
    roc_auc_true = auc(fpr_true, tpr_true)

    # 2. SALBER IPW-Adjusted ROC (from the oracle-evaluated subset in C_df)
    evaluated_samples = C_df[C_df['true_label'].notna()].copy()

    # Sort unique scores descending to sweep thresholds
    thresholds = np.sort(evaluated_samples['score'].unique())[::-1]
    tpr_ipw, fpr_ipw = [], []

    # Calculate total weighted positives and negatives in the sample
    total_pos_weight = evaluated_samples.loc[evaluated_samples['true_label'] == 1.0, 'norm_w_i'].sum()
    total_neg_weight = evaluated_samples.loc[evaluated_samples['true_label'] == 0.0, 'norm_w_i'].sum()

    for tau in thresholds:
        pred_pos = evaluated_samples['score'] >= tau
        wt_tp = evaluated_samples.loc[pred_pos & (evaluated_samples['true_label'] == 1.0), 'norm_w_i'].sum()
        wt_fp = evaluated_samples.loc[pred_pos & (evaluated_samples['true_label'] == 0.0), 'norm_w_i'].sum()

        tpr_ipw.append(wt_tp / total_pos_weight if total_pos_weight > 0 else 0)
        fpr_ipw.append(wt_fp / total_neg_weight if total_neg_weight > 0 else 0)

    # Anchor the curves at (0,0) and (1,1)
    tpr_ipw = [0.0] + tpr_ipw + [1.0]
    fpr_ipw = [0.0] + fpr_ipw + [1.0]
    roc_auc_ipw = auc(fpr_ipw, tpr_ipw)

    # 3. Plotting
    plt.figure(figsize=(4, 3))
    plt.plot(fpr_true, tpr_true, color='black', linestyle='--', linewidth=3,
             label=f'True AUC = {roc_auc_true:.3f}')
    plt.plot(fpr_ipw, tpr_ipw, color='blue', linewidth=3.5,
             label=f'SALBER AUC = {roc_auc_ipw:.3f}')

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.5)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14.5)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14.5)
    #plt.title('Validation of SALBER IPW Estimation vs. Ground Truth', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=12.5, framealpha=0.9)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"roc_{ds}.pdf", format="pdf", dpi=300)
    plt.show()



