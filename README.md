# SALBER: Sequential Active Learning Meta-Blocking for Entity Resolution

## Overview

Entity Resolution (ER) relies on blocking and meta-blocking architectures to tame its quadratic time complexity. While active learning reduces the manual labeling overhead of supervised meta-blocking, existing threshold-free and rule-based candidate pruning methods lack formal statistical guarantees. Consequently, they are vulnerable to premature termination or exhaustive labeling costs caused by the high variance of heavy-tailed candidate pair distributions.

**SALBER** is an active learning meta-blocking framework for ER that introduces a strict mathematically bounded architecture. By employing variance-bounded sampling equipped with empirical Bernstein bounds, SALBER provides rigorous statistical stopping conditions. This formulation mathematically guarantees that the active learning loop only terminates when a strict, user-defined operational recall target is met with high confidence. Comprehensive experimental evaluations across both sparse lexical and dense semantic retrieval spaces demonstrate that SALBER consistently satisfies theoretical recall guarantees. Compared to state-of-the-art candidate pruning frameworks, SALBER significantly minimizes total oracle query volume while optimizing the trade-off between labeling efficiency and recall.

## Key Features

* **Empirical Bernstein Guarantees:** Mathematically bounded stopping conditions that protect against heavy-tailed variance and prevent premature termination.
* **Operational Recall Targets:** Empowers users to strictly define minimum required recall thresholds.
* **Robust Active Learning:** Incorporates Hájek normalization and 95th-percentile weight clipping to stabilize Extreme Inverse Probability Weighting.
* **Dual Retrieval Space Support:** Evaluated across both sparse lexical (BM25) and dense semantic retrieval vector spaces (BAAI/bge-small-en-v1.5 using the HNSW ANNS index).

## Prerequisites and Installation

### 1. Git Large File Storage (LFS)

**CRITICAL:** This repository contains large benchmark datasets hosted via Git LFS. You **must** install Git LFS on your system before cloning the repository. If you clone the repository without LFS, you will only download 1KB text pointers, and the data loaders will crash.

```bash
# Install Git LFS (Ubuntu/Debian)
sudo apt update
sudo apt install git-lfs

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://anonymous.4open.science/r/SALBER-7B5E/

```

### 2. Python Dependencies

The framework requires Python 3.8+ and relies on core data science and semantic retrieval libraries.

```bash
# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

```


## Repository Structure and Usage

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

During execution, the scripts will output the progressive oracle query volume, the variance-bounded sampling metrics, and the final empirical recall confirming that the framework successfully satisfied the defined recall target.
