import pandas as pd
import numpy as np
import lib
BASE_PATH = "./data/"

amazon_df = pd.read_csv(f"{BASE_PATH}amazon_products.csv", encoding='latin1')
walmart_df = pd.read_csv(f"{BASE_PATH}walmart_products.csv", encoding='latin1')
if 'id' in amazon_df.columns:
    amazon_df = amazon_df.rename(columns={'id': 'id1'})
if 'id' in walmart_df.columns:
    walmart_df = walmart_df.rename(columns={'id': 'id2'})
truth_df = pd.read_csv(f"{BASE_PATH}truth_amazon_walmart.tsv", sep="\t", encoding="unicode_escape", keep_default_na=False)

amazon_df['id1'] = pd.to_numeric(amazon_df['id1'], errors='coerce')
amazon_df.dropna(subset=['id1'], inplace=True)
amazon_df['id1'] = amazon_df['id1'].astype(int)
walmart_df['id2'] = pd.to_numeric(walmart_df['id2'], errors='coerce')
walmart_df.dropna(subset=['id2'], inplace=True)
walmart_df['id2'] = walmart_df['id2'].astype(int)

amazon_df.reset_index(drop=True, inplace=True)
walmart_df.reset_index(drop=True, inplace=True)

truth_df.dropna(subset=["id1", "id2"], inplace=True)
truth_df["id1"] = truth_df["id1"].astype(int)
truth_df["id2"] = truth_df["id2"].astype(int)
valid_ids_a = set(amazon_df['id1'].values)
valid_ids_b = set(walmart_df['id2'].values)
truth_df = truth_df[truth_df["id2"].isin(valid_ids_b) & truth_df["id1"].isin(valid_ids_a)]
id_col_b, id_col_a = truth_df.columns[0], truth_df.columns[1]

amazon_df['text'] = (amazon_df['title'].fillna('') + ' ' + amazon_df['shortdescr'].fillna('') + ' ' + amazon_df['longdescr'].fillna('')).str.lower()
walmart_df['text'] = (walmart_df['title'].fillna('') + ' ' + walmart_df['shortdescr'].fillna('') + ' ' + walmart_df['longdescr'].fillna('')).str.lower()
ds = "Amazon-Walmart"
print(f"Aligned Table A ID: '{id_col_a}', Table B ID: '{id_col_b}'")
use_early=False
n_iterations = 10
all_metrics = []
epsilon_R = 0.1
m=200
for i in range(n_iterations):
    print(f"\n--- Starting Iteration {i+1}/{n_iterations} ---")
    np.random.seed(42 + i)
    model = lib.SALBER(
      df_a=walmart_df,
      df_b=amazon_df,
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
    # 3. Run the active learning framework
    final_tau, candidates, metrics, trajectory_df  = model.fit(seed_size=800)
    # 4. Store the results
    all_metrics.append(metrics)

# --- Compute Mean and Standard Deviation ---

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

# Optional: Save the raw multi-run data to CSV for your records
suffix = "early_ON" if use_early else "early_OFF"
safe_eps = str(epsilon_R).replace('.', '_')
df.to_csv(f"{BASE_PATH}{ds}_results_{suffix}_{safe_eps}_{m}_cosine.csv", index=False)



