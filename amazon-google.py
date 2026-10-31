import pandas as pd
import numpy as np
import lib
BASE_PATH = "./data/"

df_a = pd.read_csv(f"{BASE_PATH}Amazon.csv", encoding='latin1')
df_b = pd.read_csv(f"{BASE_PATH}GoogleProducts.csv", encoding='latin1')
truth_df = pd.read_csv(f"{BASE_PATH}truth_amazon_google.csv")
ds="Amazon-Google"


print(f"Loaded DF_A: {df_a.shape}, DF_B: {df_b.shape}, Ground Truth Matches: {truth_df.shape[0]}")
# Pre-align the primary key column names to match the ground truth file
# (e.g., if truth_df columns are ['idAbt', 'idBuy'])
id_col_a, id_col_b = truth_df.columns[0], truth_df.columns[1]

if 'id' in df_a.columns and id_col_a not in df_a.columns:
    df_a = df_a.rename(columns={'id': id_col_a})
if 'id' in df_b.columns and id_col_b not in df_b.columns:
    df_b = df_b.rename(columns={'id': id_col_b})

df_a['text'] = (df_a['name'].fillna('') + ' ' + df_a['description'].fillna('')).str.lower()
df_b['text'] = (df_b['name'].fillna('') + ' ' + df_b['description'].fillna('')).str.lower()


print(f"Aligned Table A ID: '{id_col_a}', Table B ID: '{id_col_b}'")

print("\nInitializing SALBER framework...")
use_early=False
n_iterations = 10
all_metrics = []
epsilon_R = 0.1
m=200
for i in range(n_iterations):
    print(f"\n--- Starting Iteration {i+1}/{n_iterations} ---")
    np.random.seed(42 + i)
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
      m_v=m,          # Draw more validation samples per step for faster penalty decay
      ds=ds,
      k_neighbors=3,
      use_early_stopping=use_early
    )
    # 3. Run the active learning framework
    final_tau, candidates, metrics, trajectory_df = model.fit(seed_size=800)
    # 4. Store the results
    all_metrics.append(metrics)

df = pd.DataFrame(all_metrics)
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



