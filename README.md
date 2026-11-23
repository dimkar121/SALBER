# SALBER: A Variance-Aware Active Learning Framework for Scalable Meta-Blocking

## Overview

Entity Resolution (ER) relies on blocking and meta-blocking architectures to tame its quadratic time complexity. While active learning reduces the manual labeling overhead of supervised meta-blocking, existing threshold-free and rule-based candidate pruning methods lack formal, statistical guarantees. Consequently, they remain vulnerable to premature termination or excessive labeling costs driven by the extreme class imbalance and heavy-tailed variance of ER candidate pools.

**SALBER** (Sequential Active Learning for Blocking and Entity Resolution) is a scorer-agnostic, active learning meta-blocking framework that provides **formal, finite-sample probabilistic recall guarantees**. By decoupling active threshold learning from sequential recall certification and combining **Defensive Importance Sampling** with **Time-Uniform Empirical Bernstein confidence sequences**, SALBER guarantees that the learned similarity threshold achieves a user-specified operational recall target ($1 - \epsilon_{\text{op}}$) with high probability ($1 - \delta$). Comprehensive experimental evaluations across both sparse lexical and dense semantic retrieval spaces demonstrate that SALBER consistently certifies operational recall targets while significantly reducing total oracle query volume compared to state-of-the-art candidate pruning frameworks.

## Key Features

- **Decoupled Two-Phase Architecture:** Separates *Phase 1 (Active Threshold Learning)*—which optimizes a cost-sensitive threshold $\tau^*$ via active margin sampling until stabilization—from *Phase 2 (Sequential Recall Certification)*, which freezes $\tau^*$ and certifies global recall over the residual unqueried pool.
- **Time-Uniform Empirical Bernstein Certification:** Uses polynomial failure-budget allocation ($\delta_t$) to maintain simultaneous, anytime-valid upper bounds on global false negative error and lower bounds on global match prevalence, guaranteeing recall $\ge 1 - \epsilon_{\text{op}}$ with probability $\ge 1 - \delta$.
- **Unbiased Defensive Importance Sampling:** Combines $\mathcal{O}(1)$ fixed-width score bucketing with a 50/50 defensive mixture of uniform-bucket and pool-proportional sampling, strictly bounding all IPW weights below $2$ without weight clipping or truncation bias.
- **Dual Retrieval Space Support:** Scorer-agnostic by design and evaluated across both sparse lexical (BM25) and dense semantic vector spaces (`BAAI/bge-small-en-v1.5` with HNSW ANNS indexing).

## Prerequisites and Installation

The framework requires Python 3.8+ and relies on core data science and semantic retrieval libraries.

```bash
# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

```


## Repository Structure and Usage

### Framework Classes
The framework provides two primary classes depending on the underlying similarity metric and retrieval space:

* SALBER: Designed for dense semantic vector spaces, natively supporting Inner Product similarity calculations.

* SALBER_BM25: Designed for sparse lexical retrieval, optimized for the BM25 scoring algorithm.

### Execution
The repository includes standalone Python executables tailored to standard Entity Resolution benchmark datasets. By default, the active learning loop is initialized with a seed set of 900 queries and a batch size of m = 200 queries per iteration.

To execute the SALBER framework on a specific dataset, run the corresponding script from the terminal:

* `python abt-buy.py` — Abt-Buy e-commerce dataset
* `python amazon-google.py` — Amazon-Google software dataset
* `python retailers-c.py` — RETAILERS (Clean variant) FMCG dataset
* `python retailers-d.py` — RETAILERS (Dirty variant) FMCG dataset
* `python acm-dblp.py` — ACM-DBLP citation dataset
* `python dblp-scholar.py` — DBLP-Scholar citation dataset
* `python amazon-walmart.py` — Amazon-Walmart e-commerce dataset
* `python dblp.py` — DBLP semi-synthetic citation dataset

During execution, the scripts log the phase-by-phase oracle query volume, the defensive IPW certification metrics (global false negative upper bound $U_t^R$, dynamic error tolerance $\epsilon_t^R$), certified recall, estimated recall, and global (actual) recall confirming that the certified threshold $\tau^*$ satisfies the target operational recall ($1 - \epsilon_{\text{op}}$).
