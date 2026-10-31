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

class EarlyStopper:
    """
    Implements Empirical Early Stopping (EES) based on threshold stability.
    Halts the active learning loop when the threshold stops oscillating.
    """
    def __init__(self, patience_window: int = 10, tolerance: float = 1e-5):
        """
        Args:
            patience_window (int): The number of iterations (W) to monitor for stability.
            tolerance (float): The maximum allowed difference between the min and max
                               threshold values within the window (eta).
        """
        self.patience_window = patience_window
        self.tolerance = tolerance

        # A sliding window to store the last W threshold values
        self.tau_history = deque(maxlen=patience_window)

    def update_and_check(self, current_tau: float) -> bool:
        """
        Updates the history with the new threshold and checks if the stopping
        condition is met.

        Args:
            current_tau (float): The threshold value at the current iteration.

        Returns:
            bool: True if the loop should stop, False otherwise.
        """
        self.tau_history.append(current_tau)

        # We can't evaluate stability until the window is full
        if len(self.tau_history) < self.patience_window:
            return False

        # Calculate the max difference in the trailing window
        window_max = max(self.tau_history)
        window_min = min(self.tau_history)

        # Halt if the threshold variance is below the microscopic tolerance
        if (window_max - window_min) <= self.tolerance:
            return True

        return False


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
      use_early_stopping: bool = True,
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
    # Retrieve top-k candidates from corpus_b for each query in queries_a
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
    s_min, s_max = (
        candidates['raw_score'].min(),
        candidates['raw_score'].max(),
    )
    if s_max > s_min:
      candidates['score'] = (candidates['raw_score'] - s_min) / (s_max - s_min)
    else:
      candidates['score'] = 1.0

    print(
        f'BM25 Blocking complete. Generated {len(candidates)} candidate pairs.'
    )
    candidates = candidates[candidates['score'] > self.lower_sim].reset_index(
        drop=True
    )
    print(
        f'Using {len(candidates)} candidate pairs with score > {self.lower_sim}.'
    )

    self.C = candidates.copy()
    self.N_pool = len(self.C)

    # ==========================================
    # 2. Oracle Matching Setup
    # ==========================================
    gs_a = gold_standard[self.id_col_a].astype(str)
    gs_b = gold_standard[self.id_col_b].astype(str)
    self.oracle_set = set(zip(gs_a, gs_b))

    # Diagnostic checks
    C_tuples = set(zip(self.C[self.id_col_a], self.C[self.id_col_b]))
    self.recalled_matches = len(self.oracle_set.intersection(C_tuples))
    print(
        f'\n[DIAGNOSTIC] Total True matches in Gold Standard:'
        f' {len(self.oracle_set)}'
    )
    print(
        '[DIAGNOSTIC] True matches successfully retrieved by BM25:'
        f' {self.recalled_matches}\n'
    )
    print(f'[TARGETS] Target Conditional Recall: >= {1.0 - epsilon_R:.2f}')

    self.eps_P = epsilon_P
    self.epsilon_op = epsilon_R
    self.eps_R = epsilon_R
    self.delta = delta
    self.eta = eta
    self.m = m
    self.m_v = m_v
    self.use_early_stopping = use_early_stopping
    self.V_samples = []

    # ==========================================
    # 3. Similarity Buckets & IPW Weights
    # ==========================================
    self.C['bucket'] = pd.cut(self.C['score'], bins=K, labels=False)
    unique_buckets = self.C['bucket'].unique()
    self.bucket_probs = {b: 1.0 / len(unique_buckets) for b in unique_buckets}

    bucket_sizes = self.C['bucket'].value_counts().to_dict()
    self.C['q_i'] = self.C['bucket'].map(
        lambda b: self.bucket_probs[b] / bucket_sizes[b]
    )
    self.C['norm_w_i'] = 1.0 / (self.N_pool * self.C['q_i'])

    self.c = float(np.percentile(self.C['norm_w_i'].values, 85.0))

    self.C['in_D'] = False
    self.C['in_V'] = False
    self.C['true_label'] = np.nan

    self.tau = 0.5
    self.t = 0

  def _query_oracle(self, idx_list):
    pairs_A = self.C.loc[idx_list, self.id_col_a].values
    pairs_B = self.C.loc[idx_list, self.id_col_b].values
    labels = [
        1.0 if pair in self.oracle_set else 0.0
        for pair in zip(pairs_A, pairs_B)
    ]
    self.C.loc[idx_list, 'true_label'] = labels

  def _retrain_hypothesis(self) -> float:
    D = self.C[self.C['in_D']]
    if len(D) == 0:
      return self.tau

    scores = D['score'].values
    labels = D['true_label'].values

    # Strict Clipping using the global constant
    base_weights = np.clip(D['norm_w_i'].values, a_min=0.0, a_max=self.c)

    # Hájek Normalization
    base_weights = base_weights * (len(D) / np.sum(base_weights))

    # Dynamic Balancing Ratio (B_t)
    pos_mask_weights = labels == 1.0
    neg_mask_weights = labels == 0.0

    pos_weight_sum = np.sum(base_weights[pos_mask_weights])
    neg_weight_sum = np.sum(base_weights[neg_mask_weights])

    if pos_weight_sum > 0:
      dynamic_ratio = 8.0 * (neg_weight_sum / pos_weight_sum)
    else:
      dynamic_ratio = 1.0

    # Vectorized Threshold Solver
    candidate_taus = np.unique(scores)
    preds = scores[None, :] >= candidate_taus[:, None]

    fn_mask = (labels == 1.0) & ~preds
    fp_mask = (labels == 0.0) & preds

    fn_cost = np.sum(fn_mask * base_weights, axis=1) * dynamic_ratio
    fp_cost = np.sum(fp_mask * base_weights, axis=1)

    total_weighted_error = fn_cost + fp_cost
    best_idx = np.argmin(total_weighted_error)
    return float(candidate_taus[best_idx])

  def _calculate_bernstein_bound(
      self, loss_array: np.ndarray, N_v: int
  ) -> float:
    if N_v == 0:
      return float('inf')

    E_hat = np.mean(loss_array)
    V_hat = np.var(loss_array, ddof=0)

    delta_t = (6.0 * self.delta) / ((np.pi**2) * (self.t**2))

    variance_penalty = np.sqrt((2.0 * V_hat * np.log(1.0 / delta_t)) / N_v)
    max_possible_loss = self.c * (1.0 - self.epsilon_op)
    scale_penalty = (3.0 * max_possible_loss * np.log(1.0 / delta_t)) / N_v

    total_penalty = variance_penalty + scale_penalty

    if N_v < self.N_pool:
      fpc = np.sqrt(max(0.0, (self.N_pool - N_v) / (self.N_pool - 1.0)))
    else:
      fpc = 0.0

    return E_hat + (total_penalty * fpc)

  def fit(self, seed_size: int = 200) -> Tuple[float, pd.DataFrame, dict]:
    time_start = time.time()
    p_seed = self.C['q_i'].values
    p_seed = p_seed / p_seed.sum()

    seed_idx = np.random.choice(
        self.C.index, size=seed_size, replace=False, p=p_seed
    )
    self.C.loc[seed_idx, 'in_D'] = True
    self._query_oracle(seed_idx)
    self.tau = self._retrain_hypothesis()
    print(f'Initialization complete. Starting tau_1 = {self.tau:.4f}')

    early_stopper = EarlyStopper(patience_window=15, tolerance=1e-3)
    current_estimated_recall = 0.0
    ess = 0.0

    while True:
      self.t += 1

      # 1. Validation Draw (V)
      p_dist = self.C['q_i'].values
      p_dist = p_dist / p_dist.sum()
      v_draws = np.random.choice(
          self.C.index, size=self.m_v, replace=True, p=p_dist
      )
      self.V_samples.extend(v_draws)

      unlabeled_v = [
          idx
          for idx in v_draws
          if pd.isna(self.C.loc[idx, 'true_label'])
      ]
      if unlabeled_v:
        self._query_oracle(unlabeled_v)

      # 2. Active Training Margin Draw (D)
      margin_mask = (np.abs(self.C['score'] - self.tau) <= self.eta) & (
          ~self.C['in_D']
      )
      margin_candidates = self.C[margin_mask].index

      if len(margin_candidates) > 0:
        draw_b = min(self.m, len(margin_candidates))
        d_draws = np.random.choice(
            margin_candidates, size=draw_b, replace=False
        )
        self.C.loc[d_draws, 'in_D'] = True

        unlabeled_d = [
            idx
            for idx in d_draws
            if pd.isna(self.C.loc[idx, 'true_label'])
        ]
        if unlabeled_d:
          self._query_oracle(unlabeled_d)

      # 3. Retrain Hypothesis
      self.tau = self._retrain_hypothesis()

      # 4. Retroactive Validation Sweep
      V = self.C.loc[self.V_samples]
      N_v = len(V)

      if N_v > 0:
        preds = (V['score'].values >= self.tau).astype(float)
        labels = V['true_label'].values

        clipped_weights = np.clip(
            V['norm_w_i'].values, a_min=0.0, a_max=self.c
        )
        weights = clipped_weights * (N_v / np.sum(clipped_weights))
        ess = (
            (np.sum(weights) ** 2) / np.sum(weights**2)
            if np.sum(weights) > 0
            else 0.0
        )

        loss_P_array = (
            weights * ((labels == 0.0) & (preds == 1.0)).astype(float)
        )
        loss_R_array = (
            weights * ((labels == 1.0) & (preds == 0.0)).astype(float)
        )

        true_E_R = np.mean(loss_R_array)
        U_R = self._calculate_bernstein_bound(loss_R_array, N_v)

        est_TP = np.sum(
            weights * ((labels == 1.0) & (preds == 1.0)).astype(float)
        )
        est_FN = np.sum(loss_R_array)
        current_estimated_recall = (
            est_TP / (est_TP + est_FN) if (est_TP + est_FN) > 0 else 0.0
        )

        allowed_fn_ratio = self.epsilon_op / (1.0 - self.epsilon_op)
        allowed_false_negatives = self.recalled_matches * allowed_fn_ratio
        dynamic_goal = allowed_false_negatives / self.N_pool
        N_total_unique = self.C['true_label'].notna().sum()

        if self.t % 5 == 0:
          matches_found = int((self.C['true_label'] == 1.0).sum())
          print(f'Iter {self.t:3d} (N_v={N_v:5d}) | tau: {self.tau:.4f}')
          print(
              f'  [MATH] FNR -> True Error: {true_E_R:.6f} | Bound U^R:'
              f' {U_R:.6f} (Goal: <= {dynamic_goal:.6f})'
          )
          print(
              '  [MATH] Est. Recall:'
              f' {current_estimated_recall:.4f} | Bound U^R: {U_R:.6f} (Goal: <='
              f' {dynamic_goal:.6f})'
          )
          print(
              f'  [ORACLE STATS] Total pairs submitted: {N_total_unique:,} |'
              f' True matches found: {matches_found}'
          )

        if U_R <= dynamic_goal:
          matches_found = int((self.C['true_label'] == 1.0).sum())
          print(f'\n[CONVERGENCE] Certified safely at iteration {self.t}.')
          print(
              f'  [ORACLE STATS] Total pairs submitted: {N_total_unique:,} |'
              f' True matches found: {matches_found}'
          )
          print(
              f'Recall Target Met. Final Bound {U_R:.6f} <= Goal'
              f' {dynamic_goal:.6f}'
          )
          break

        if self.use_early_stopping:
          heuristic_satisfied = early_stopper.update_and_check(self.tau)
          if heuristic_satisfied:
            print(
                f'Iter {self.t}: [STOP] Empirical Early Stopping. Threshold'
                f' stabilized at {self.tau:.4f}.'
            )
            break

      if self.t >= 1000:
        print('\n[TIMEOUT] Reached maximum allowed iterations.')
        break

    # Unseen Evaluation
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
      print(
          f'\n--- Final Evaluation on Unseen Candidates ({len(unseen):,}) ---'
      )
      print(
          f'True Positives: {TP} | False Positives: {FP} | False Negatives: {FN}'
      )
      print(f'Precision: {prec:.4f} | Recall: {rec:.4f}')
    else:
      prec, rec = 0.0, 0.0

    total_time = time.time() - time_start
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    metrics = {
        'total_time': total_time,
        'estimated_recall': current_estimated_recall,
        'actual_recall': rec,
        'actual_precision': prec,
        'f1': f1,
        'ess': ess,
        'queries': int(self.C['true_label'].notna().sum()),
        'positive_yield': int((self.C['true_label'] == 1.0).sum()),
    }
    return self.tau, self.C, metrics


class SALBER:
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
                 use_early_stopping:bool=True,
                 emb_dir:str="./data/",
                 lower_sim:float=0.4):

        self.model = SentenceTransformer(model_name)
        # ==========================================
        # 1. Embedding Generation & HNSW Blocking
        # ==========================================
        file_emb_a = f"embeddings_{ds}_a.npy"
        file_emb_b = f"embeddings_{ds}_b.npy"
        # Check and load or encode for dataset A
        if os.path.exists(f"{emb_dir}{file_emb_a}"):
           print("Loading existing embeddings ...")
           emb_a = np.load(f"{emb_dir}{file_emb_a}")
           emb_b = np.load(f"{emb_dir}{file_emb_b}")
        else:
           print("Encoding datasets ...")
           emb_a = self.model.encode(df_a['text'].tolist(), normalize_embeddings=True)
           emb_b = self.model.encode(df_b['text'].tolist(), normalize_embeddings=True)
           print("Saving datasets ...")
           np.save(f"{emb_dir}{file_emb_a}", emb_a)
           np.save(f"{emb_dir}{file_emb_b}", emb_b)

        print("Building HNSW graph index...")
        d = emb_b.shape[1]
        M = 32
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(emb_b)

        print("Querying HNSW index for top-k candidates...")
        self.k_neighbors = k_neighbors
        index.hnsw.efSearch = 128

        similarities, neighbor_indices = index.search(emb_a, k_neighbors)

        print("Formulating candidate pairs dataframe...")

        if id_col_a in df_a.columns:
            a_ids = df_a[id_col_a].astype(str).values
        else:
            a_ids = df_a.index.astype(str).values # Fallback if ID is the index

        if id_col_b in df_b.columns:
            b_ids = df_b[id_col_b].astype(str).values
        else:
            b_ids = df_b.index.astype(str).values # Fallback if ID is the index

        A_col = np.repeat(a_ids, k_neighbors)
        B_col = b_ids[neighbor_indices.ravel()]
        score_col = similarities.ravel()

        candidates = pd.DataFrame({
            id_col_a: A_col,
            id_col_b: B_col,
            'score': score_col
        })
        self.lower_sim = lower_sim
        # Standard threshold filter
        print(f"HNSW Blocking complete. Generated {len(candidates)} candidate pairs.")
        candidates = candidates[candidates['score'] > self.lower_sim].reset_index(drop=True)
        print(f"Using {len(candidates)} candidate pairs lower sim={self.lower_sim}.")

        self.C = candidates.copy()
        self.N_pool = len(self.C)
        self.id_col_a = id_col_a
        self.id_col_b = id_col_b


        # ==========================================
        # 2. Oracle Matching
        # ==========================================
        # Force gold standard to strings and create Oracle
        gs_a = gold_standard[self.id_col_a].astype(str)
        gs_b = gold_standard[self.id_col_b].astype(str)
        self.oracle_set = set(zip(gs_a, gs_b))

        # --- THE DIAGNOSTIC CHECK ---
        C_tuples = set(zip(self.C[self.id_col_a], self.C[self.id_col_b]))
        self.recalled_matches = len(self.oracle_set.intersection(C_tuples))
        print(f"\n[DIAGNOSTIC] Total True matches in Gold Standard: {len(self.oracle_set)}")
        print(f"[DIAGNOSTIC] True matches successfully retrieved by HNSW: {self.recalled_matches}\n")

        print(f"[TARGETS] Target Conditional Recall: >= {1.0 - epsilon_R:.2f}")

        self.eps_P = epsilon_P
        self.epsilon_op = epsilon_R
        self.eps_R = epsilon_R
        self.delta = delta
        self.eta = eta
        self.m = m
        self.m_v = m_v
        self.use_early_stopping = use_early_stopping
        self.V_samples = []

        # 2. Similarity Buckets & IPW Weights
        self.C['bucket'] = pd.cut(self.C['score'], bins=K, labels=False)
        unique_buckets = self.C['bucket'].unique()
        self.bucket_probs = {b: 1.0 / len(unique_buckets) for b in unique_buckets}

        bucket_sizes = self.C['bucket'].value_counts().to_dict()
        self.C['q_i'] = self.C['bucket'].map(lambda b: self.bucket_probs[b] / bucket_sizes[b])
        self.C['norm_w_i'] = 1.0 / (self.N_pool * self.C['q_i'])

        self.c = np.percentile(self.C['norm_w_i'].values, 85.0)

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
        # 1. Get the Active Training Set
        D = self.C[self.C['in_D']]
        if len(D) == 0:
            return self.tau # Fallback if empty

        scores = D['score'].values
        labels = D['true_label'].values

        # 2. Strict Clipping using the global constant (self.c)
        base_weights = np.clip(D['norm_w_i'].values, a_min=0.0, a_max=self.c)

        # 3. Hájek Normalization (Keep the distribution sound before altering)
        base_weights = base_weights * (len(D) / np.sum(base_weights))

        # 4. Calculate the Dynamic Balancing Ratio (B_t)
        pos_mask_weights = (labels == 1.0)
        neg_mask_weights = (labels == 0.0)

        pos_weight_sum = np.sum(base_weights[pos_mask_weights])
        neg_weight_sum = np.sum(base_weights[neg_mask_weights])

        if pos_weight_sum > 0:
            dynamic_ratio = 8 * (neg_weight_sum / pos_weight_sum)
            # print(f"NEG/POS {dynamic_ratio}") # Uncomment to track the multiplier
        else:
            dynamic_ratio = 1.0 # Default safely if no positives exist yet

        # -------------------------------------------------------------
        # 5. THE VECTORIZED THRESHOLD SOLVER (Direct mapping to Eq. 1 with B_t)
        # -------------------------------------------------------------
        candidate_taus = np.unique(scores)

        # h_tau(s_i): Predict 1 if score >= candidate, else 0
        # Broadcasting: Compares ALL candidates against ALL scores simultaneously
        # Shape of preds: (num_candidates, num_samples)
        preds = scores[None, :] >= candidate_taus[:, None]

        # I[Y_i = 1 AND h_tau(s_i) = 0] -> False Negatives matrix
        fn_mask = (labels == 1.0) & ~preds

        # I[Y_i = 0 AND h_tau(s_i) = 1] -> False Positives matrix
        fp_mask = (labels == 0.0) & preds

        # \sum w_i * ( B_t * \lambda * FN + FP )
        # Sum across the sample dimension (axis=1) for every candidate threshold
        # We apply BOTH the dynamic ratio and the static lambda penalty to the FNs
        fn_cost = np.sum(fn_mask * base_weights, axis=1) * dynamic_ratio
        fp_cost = np.sum(fp_mask * base_weights, axis=1)

        # Total cost for each candidate threshold
        total_weighted_error = fn_cost + fp_cost

        # Find the index of the threshold that yields the absolute minimum cost
        best_idx = np.argmin(total_weighted_error)
        best_tau = candidate_taus[best_idx]

        return float(best_tau)


    def _calculate_bernstein_bound(self, loss_array: np.ndarray, N_v: int) -> float:
      if N_v == 0:
        return float('inf')

      E_hat = np.mean(loss_array)
      V_hat = np.var(loss_array, ddof=0)

      delta_t = (6.0 * self.delta) / ((np.pi ** 2) * (self.t ** 2))

      # 1. Variance Penalty
      variance_penalty = np.sqrt((2.0 * V_hat * np.log(1.0 / delta_t)) / N_v)

      # 2. Scale Penalty (Updated for Asymmetric Loss Range)
      # The absolute maximum possible value is c * (1 - epsilon_op)
      max_possible_loss = self.c * (1.0 - self.epsilon_op)
      scale_penalty = (3.0 * max_possible_loss * np.log(1.0 / delta_t)) / N_v

      total_penalty = variance_penalty + scale_penalty

      if N_v < self.N_pool:
        fpc = np.sqrt(max(0.0, (self.N_pool - N_v) / (self.N_pool - 1.0)))
      else:
        fpc = 0.0

      return E_hat + (total_penalty * fpc)




    def fit(self, seed_size: int = 200) -> float:
        # --- Seed initialization ---
        # 1. Normalize the existing probability distribution for the entire pool
        time_start = time.time()
        p_seed = self.C['q_i'].values
        p_seed = p_seed / p_seed.sum()

        # 2. Draw the seed using your bucket-balanced distribution (q_i)
        seed_idx = np.random.choice(self.C.index, size=seed_size, replace=False, p=p_seed)

        self.C.loc[seed_idx, 'in_D'] = True
        self._query_oracle(seed_idx)
        self.tau = self._retrain_hypothesis()
        print(f"Initialization complete. Starting tau_1 = {self.tau:.4f}")

        early_stopper = EarlyStopper(patience_window=15, tolerance=1e-3)
        #Initialize the trajectory logger
        self.trajectory_log = []
        while True:
            self.t += 1

            # 1. Draw a batch for Validation (V)
            # Exclude anything that has already been labeled for V OR D
            # We draw from the entire pool, strictly using the IPW distribution
            p_dist = self.C['q_i'].values
            p_dist = p_dist / p_dist.sum()
            # Notice: replace=True and we append to a running list of samples
            v_draws = np.random.choice(self.C.index, size=self.m_v, replace=True, p=p_dist)
            self.V_samples.extend(v_draws) # self.V_samples is initialized as [] in __init__

            # Check the Global Cache: Only query the Oracle for pairs that don't have a label yet
            unlabeled_v = [idx for idx in v_draws if pd.isna(self.C.loc[idx, 'true_label'])]
            if unlabeled_v:
                  self._query_oracle(unlabeled_v)
            # 2. Query the margin for Active Training Set (D)
            margin_mask = (np.abs(self.C['score'] - self.tau) <= self.eta) & (~self.C['in_D'])
            margin_candidates = self.C[margin_mask].index

            if len(margin_candidates) > 0:
                draw_b = min(self.m, len(margin_candidates))
                d_draws = np.random.choice(margin_candidates, size=draw_b, replace=False)
                self.C.loc[d_draws, 'in_D'] = True # Add them to the training mask

                # Check the Global Cache: Only query the Oracle for unknown labels
                # (Some might have been randomly labeled by V earlier!)
                unlabeled_d = [idx for idx in d_draws if pd.isna(self.C.loc[idx, 'true_label'])]
                if unlabeled_d:
                  self._query_oracle(unlabeled_d)

            # 3. Retrain
            self.tau = self._retrain_hypothesis()

            # 4. Retroactive Validation Sweep
            #V = self.C[self.C['in_V']]
            V = self.C.loc[self.V_samples]
            N_v = len(V)

            if N_v > 0:
                preds = (V['score'].values >= self.tau).astype(float)
                labels = V['true_label'].values

                # --- AGGRESSIVE Weight Clipping ---
                # --- AGGRESSIVE Weight Clipping ---
                clipped_weights = np.clip(V['norm_w_i'].values, a_min=0.0, a_max=self.c)

                # --- Hájek Normalization ---
                weights = clipped_weights * (N_v / np.sum(clipped_weights))
                ess = (np.sum(weights)**2) / np.sum(weights**2) if np.sum(weights) > 0 else 0.0
                # Global False Positives
                loss_P_array = weights * ((labels == 0.0) & (preds == 1.0)).astype(float)

                # Global False Negatives (Standard Unbiased Estimation)
                loss_R_array = weights * ((labels == 1.0) & (preds == 0.0)).astype(float)
                # True Errors
                true_E_P = np.mean(loss_P_array)
                true_E_R = np.mean(loss_R_array)
                # Evaluate Bounds (Make sure _calculate_bernstein_bound uses self.c in the scale penalty!)
                U_P = self._calculate_bernstein_bound(loss_P_array, N_v)
                U_R = self._calculate_bernstein_bound(loss_R_array, N_v)
                # -------------------------------------------------------------
                # CALCULATE IPW-ESTIMATED RECALL
                # -------------------------------------------------------------
                est_TP = np.sum(weights * ((labels == 1.0) & (preds == 1.0)).astype(float))
                est_FN = np.sum(loss_R_array) # Which is weights * False Negatives

                current_estimated_recall = est_TP / (est_TP + est_FN) if (est_TP + est_FN) > 0 else 0.0
                # -------------------------------------------------------------
                # THE OBSERVABLE LOWER BOUND TARGET (NO ORACLE REQUIRED)
                # -------------------------------------------------------------
                # For 80% recall (epsilon_op = 0.2), allowed FNs = (0.2 / 0.8) * TP
                # We use the matches found *so far* as a strictly conservative lower bound for TP.
                allowed_fn_ratio = self.epsilon_op / (1.0 - self.epsilon_op)
                allowed_false_negatives = self.recalled_matches * allowed_fn_ratio

                # Convert the volume to a global proportion for the Bernstein bound
                dynamic_goal = allowed_false_negatives / self.N_pool

                #N_v = self.C['in_V'].sum()
                #N_d = self.C['in_D'].sum()
                N_total_unique = self.C['true_label'].notna().sum()

                # -------------------------------------------------------------
                # LOGGING FOR CONVERGENCE PLOT
                # -------------------------------------------------------------
                # 1. Calculate Actual Recall over the ENTIRE pool using gold labels
                # (Assuming you have a 'gold_label' column with the hidden ground truth)
                global_preds = (self.C['score'].values >= self.tau).astype(float)
                total_actual_positives = (self.C['true_label'].values == 1.0).sum()
                true_positives_at_tau = ((self.C['true_label'].values == 1.0) & (global_preds == 1.0)).sum()

                current_actual_recall = true_positives_at_tau / total_actual_positives if total_actual_positives > 0 else 0.0

                # 2. Append the current state to the log
                self.trajectory_log.append({
                    'iteration': self.t,
                    'queries': int(N_total_unique),
                    'estimated_recall': current_estimated_recall,
                    'actual_recall': current_actual_recall,
                    'error_bound': U_R,
                    'tau': self.tau
                })
                # -------------------------------------------------------------




                # Print the diagnostics
                if self.t % 5 == 0:
                    matches_found = int((self.C['true_label'] == 1.0).sum())
                    print(f"Iter {self.t:3d} (N_v={N_v:5d}) | tau: {self.tau:.4f}")
                    print(f"  [MATH] FNR -> True Error: {true_E_R:.6f} | Bound U^R: {U_R:.6f} (Goal: <= {dynamic_goal:.6f})")
                    print(f"  [MATH] Est. Recall: {current_estimated_recall:.4f} | Bound U^R: {U_R:.6f} (Goal: <= {dynamic_goal:.6f})")
                    print(f"  [ORACLE STATS] Total pairs submitted: {N_total_unique:,} | True matches found: {matches_found}")



                # The Observable Stopping Condition
                if U_R <= dynamic_goal:
                    matches_found = int((self.C['true_label'] == 1.0).sum())
                    print(f"\n[CONVERGENCE] Certified safely at iteration {self.t}.")

                    print(f"  [ORACLE STATS] Total pairs submitted: {N_total_unique:,} | True matches found: {matches_found}")
                    print(f"Recall Target Met. Final Bound {U_R:.6f} <= Goal {dynamic_goal:.6f}")
                    break
                if self.use_early_stopping:
                  heuristic_satisfied = early_stopper.update_and_check(self.tau)
                  if heuristic_satisfied:
                     print(f"Iter {self.t}: [STOP] Empirical Early Stopping. Threshold stabilized at {self.tau:.4f}.")
                     break
            if self.t >= 1000:
                print("\n[TIMEOUT] Reached maximum allowed iterations.")
                break

        #Save the trajectory to a CSV before returning
        trajectory_df = pd.DataFrame(self.trajectory_log)

        # Final Unseen Evaluation
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
        time_end = time.time()
        total_time = time_end - time_start
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        metrics = {
          'total_time': total_time,
          'estimated_recall': current_estimated_recall,
          'actual_recall': rec,
          'actual_precision': prec,
          'f1': f1,
          'ess' : ess,
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



