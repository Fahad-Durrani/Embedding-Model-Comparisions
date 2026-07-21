"""Central configuration for the embedding-model comparison harness.

Everything that affects a run (which models, how they are prefixed, which k
values, the threshold sweep, and where files live) is declared here so runs are
reproducible and easy to tweak.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
AGGREGATE_CSV = os.path.join(RESULTS_DIR, "aggregate.csv")
PER_QUERY_CSV = os.path.join(RESULTS_DIR, "per_query.csv")
REPORT_HTML = os.path.join(RESULTS_DIR, "report.html")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Models under test
# --------------------------------------------------------------------------- #
# `query_prefix` / `passage_prefix` implement each model's own convention.
# intfloat/e5 REQUIRES "query: " / "passage: " prefixes; all-MiniLM uses none.
# Getting this wrong is the #1 way people unfairly cripple e5.
MODELS = [
    {
        "key": "minilm",
        "name": "all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 384,
        "params": "~22M",
    },
    {
        "key": "e5",
        "name": "multilingual-e5-small",
        "hf_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dim": 384,
        "params": "~118M",
    },
]

# --------------------------------------------------------------------------- #
# Retrieval / metrics
# --------------------------------------------------------------------------- #
K_VALUES = [1, 3, 5, 10, 20, 40]     # k grid for Recall/Precision/nDCG/Hit@k
NDCG_K = 10                          # headline nDCG cutoff used in summaries

# Relevance grade (from qrels) at or above which a passage counts as "relevant"
# for the binary metrics (precision/recall/hit/MRR). nDCG uses the raw grades.
RELEVANT_GRADE_THRESHOLD = 1         # grades: 2 = highly, 1 = partially, 0 = not

# --------------------------------------------------------------------------- #
# Threshold calibration & dynamic-k policy
# --------------------------------------------------------------------------- #
# Cosine-cutoff sweep grid (inclusive of start, exclusive style handled in code).
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

# Operating-point targets reported per model alongside the F1-optimal cutoff.
PRECISION_TARGET = 0.90              # smallest tau reaching this precision
RECALL_TARGET = 0.80                # largest tau still reaching this recall

# Dynamic-k retrieval policy band: keep chunks with score >= tau_model, clamped
# into [K_MIN, K_MAX]. Floor avoids starving the LLM; cap protects context/cost.
K_MIN = 5
K_MAX = 40
