import pandas as pd
import numpy as np
import lib
BASE_PATH = "./data/"


df_a = pd.read_csv(f"{BASE_PATH}DBLP.csv",  encoding='latin1')
df_b = pd.read_csv(f"{BASE_PATH}Scholar.csv",   encoding='latin1')
truth_df = pd.read_csv(f"{BASE_PATH}truth_Scholar_DBLP.csv",  encoding='latin1')
ds = "Scholar-DBLP"


print(f"Loaded DF_A: {df_a.shape}, DF_B: {df_b.shape}, Ground Truth Matches: {truth_df.shape[0]}")
id_col_a, id_col_b = truth_df.columns[0], truth_df.columns[1]

if 'id' in df_a.columns and id_col_a not in df_a.columns:
    df_a = df_a.rename(columns={'id': id_col_a})
if 'id' in df_b.columns and id_col_b not in df_b.columns:
    df_b = df_b.rename(columns={'id': id_col_b})

df_a['text'] = (df_a['title'].fillna('') + ' ' + df_a['authors'].fillna('')+' ' + df_a['venue'].fillna('')).str.lower()
df_b['text'] = (df_b['title'].fillna('') + ' ' + df_b['authors'].fillna('')+' ' + df_b['venue'].fillna('')).str.lower()

use_early=False
n_iterations = 10
all_metrics = []
epsilon_R = 0.1
m=200
for i in range(n_iterations):
    print(f"\n--- Starting Iteration {i+1}/{n_iterations} ---")
    model = lib.SALBER(
      df_a=df_a,
      df_b=df_b,
      gold_standard=truth_df,
      id_col_a=id_col_a,
      id_col_b=id_col_b,
      model_name = 'BAAI/bge-small-en-v1.5',
      epsilon_R=epsilon_R,
      delta =  0.1,
      m=m,
      m_v=m,
      ds=ds,
      k_neighbors=3,
      use_early_stopping=use_early
    )
    np.random.seed(42 + i)
    # 3. Run the active learning framework
    final_tau, candidates, metrics = model.fit(seed_size=800)
    # 4. Store the results
    all_metrics.append(metrics)
# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(all_metrics)
# Calculate and print Mean ± SD for every metric
for column in df.columns:
    # Skip any non-numeric columns if you added dataset names
    if pd.api.types.is_numeric_dtype(df[column]):
        mean_val = df[column].mean()
        sd_val = df[column].std()
        # Format: e.g., Recall: 0.9621 ± 0.0142
        #print(f"{column}: {mean_val:.4f} ± {sd_val:.4f}")

suffix = "early_ON" if use_early else "early_OFF"
safe_eps = str(epsilon_R).replace('.', '_')
df.to_csv(f"{BASE_PATH}{ds}_results_{suffix}_{safe_eps}_{m}_cosine.csv", index=False)


