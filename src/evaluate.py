"""Main evaluation runner.

Loads the dataset, runs each model end-to-end (encode -> retrieve -> metrics ->
threshold calibration -> dynamic-k simulation), and writes machine-readable
results (results.json) plus tidy CSVs for aggregate metrics and per-query
side-by-side. Run report.py afterwards to build the HTML report.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # config
sys.path.insert(0, HERE)   # sibling modules

import config  # noqa: E402
from embedder import Embedder  # noqa: E402
from retriever import Retriever  # noqa: E402
import metrics as M  # noqa: E402
from calibrate import calibrate_model  # noqa: E402

TOP_DISPLAY = 10  # ranked rows stored per query for the side-by-side deep dive


def load_dataset() -> dict:
    with open(config.DATASET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def run_model(cfg: dict, dataset: dict) -> dict:
    passages = dataset["passages"]
    queries = dataset["queries"]
    passage_ids = [p["id"] for p in passages]
    passage_text = {p["id"]: p["text"] for p in passages}

    print(f"  [{cfg['name']}] loading model ...")
    emb = Embedder(cfg).load()

    print(f"  [{cfg['name']}] encoding {len(passages)} passages + {len(queries)} queries ...")
    p_emb = emb.encode_passages([p["text"] for p in passages])
    q_emb = emb.encode_queries([q["text"] for q in queries])

    retriever = Retriever(passage_ids, p_emb)
    ranked_lists = retriever.rank(q_emb)  # aligned with queries order

    ranked_by_query = {q["id"]: ranked_lists[i] for i, q in enumerate(queries)}
    qrels_by_query = {q["id"]: q["qrels"] for q in queries}
    categories = {q["id"]: q["category"] for q in queries}

    # ---- per-query + aggregate metrics ------------------------------------
    per_query = {}
    for q in queries:
        ranked = ranked_by_query[q["id"]]
        pq = M.per_query_metrics(ranked, q["qrels"], config.K_VALUES, config.RELEVANT_GRADE_THRESHOLD)
        pq["ranked_top"] = [
            {"id": pid, "score": round(s, 4), "grade": q["qrels"].get(pid, 0)}
            for pid, s in ranked[:TOP_DISPLAY]
        ]
        per_query[q["id"]] = pq

    aggregate = M.aggregate(per_query, config.K_VALUES)
    by_category = M.aggregate_by_category(per_query, categories, config.K_VALUES)

    # score distributions for the report histograms
    rel_scores, irr_scores, nomatch_max = [], [], []
    for q in queries:
        ranked = ranked_by_query[q["id"]]
        rel = M.relevant_set(q["qrels"], config.RELEVANT_GRADE_THRESHOLD)
        for pid, s in ranked:
            (rel_scores if pid in rel else irr_scores).append(round(s, 4))
        if not rel and ranked:
            nomatch_max.append(round(ranked[0][1], 4))
    score_dist = {"relevant": rel_scores, "irrelevant": irr_scores, "nomatch_max": nomatch_max}

    # ---- threshold calibration + dynamic-k --------------------------------
    print(f"  [{cfg['name']}] calibrating threshold + simulating dynamic-k ...")
    calibration = calibrate_model(
        ranked_by_query, qrels_by_query, config.RELEVANT_GRADE_THRESHOLD,
        config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEP,
        config.PRECISION_TARGET, config.RECALL_TARGET, config.K_MIN, config.K_MAX,
    )

    return {
        "key": cfg["key"],
        "name": cfg["name"],
        "hf_id": cfg["hf_id"],
        "params": cfg["params"],
        "dim": cfg["dim"],
        "query_prefix": cfg["query_prefix"],
        "passage_prefix": cfg["passage_prefix"],
        "timing": emb.timing.as_dict(),
        "aggregate": aggregate,
        "by_category": by_category,
        "per_query": per_query,
        "score_dist": score_dist,
        "calibration": calibration,
    }


def _sanity_checks(results: dict) -> None:
    problems = []
    for key, m in results["models"].items():
        # scores in [-1, 1]
        for qid, pq in m["per_query"].items():
            for row in pq["ranked_top"]:
                if not (-1.01 <= row["score"] <= 1.01):
                    problems.append(f"{key}/{qid}: score {row['score']} out of range")
        # two models should end up with different F1-optimal taus (per-model calibration)
    taus = {k: m["calibration"]["operating_points"]["f1_optimal"]["tau"] for k, m in results["models"].items()}
    print(f"  F1-optimal tau per model: {taus}")
    if len(set(taus.values())) == 1:
        print("  NOTE: models share the same F1-optimal tau on this dataset (still calibrated independently).")
    if problems:
        raise AssertionError("sanity checks failed:\n" + "\n".join(problems[:10]))
    # exact_paraphrase gold@1 for at least one model
    ep = [q["id"] for q in results["dataset"]["queries"] if q["category"] == "exact_paraphrase"]
    for qid in ep:
        ranks = [results["models"][k]["per_query"][qid]["gold_rank"] for k in results["models"]]
        if not any(r == 1 for r in ranks):
            print(f"  WARN: no model ranked gold #1 for exact_paraphrase {qid} (ranks={ranks})")


def write_csvs(results: dict) -> None:
    # tidy aggregate metrics: model, metric, k, value
    rows = []
    for key, m in results["models"].items():
        for metric_key, val in m["aggregate"].items():
            if "@" in metric_key:
                name, k = metric_key.split("@")
                k = int(k)
            else:
                name, k = metric_key, None
            rows.append({"model": m["name"], "metric": name, "k": k, "value": val})
    pd.DataFrame(rows).to_csv(config.AGGREGATE_CSV, index=False)

    # per-query side-by-side (one row per query x model)
    q_by_id = {q["id"]: q for q in results["dataset"]["queries"]}
    rows = []
    for key, m in results["models"].items():
        for qid, pq in m["per_query"].items():
            q = q_by_id[qid]
            top1 = pq["ranked_top"][0] if pq["ranked_top"] else {"id": None, "score": None}
            rows.append({
                "qid": qid, "category": q["category"], "query": q["text"],
                "model": m["name"], "gold_rank": pq["gold_rank"],
                "top1_id": top1["id"], "top1_score": top1["score"],
                "recall@10": pq["recall@10"], "ndcg@10": pq["ndcg@10"], "mrr": pq["mrr"],
                "top_score": pq["top_score"], "separability": pq["separability"],
            })
    df = pd.DataFrame(rows).sort_values(["qid", "model"])
    df.to_csv(config.PER_QUERY_CSV, index=False)


def main() -> int:
    np.random.seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)

    dataset = load_dataset()
    print(f"Dataset: {dataset['meta']['n_passages']} passages, {dataset['meta']['n_queries']} queries")

    model_results = {}
    for cfg in config.MODELS:
        model_results[cfg["key"]] = run_model(cfg, dataset)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "k_values": config.K_VALUES,
            "ndcg_k": config.NDCG_K,
            "relevant_grade_threshold": config.RELEVANT_GRADE_THRESHOLD,
            "threshold_sweep": {"min": config.THRESHOLD_MIN, "max": config.THRESHOLD_MAX, "step": config.THRESHOLD_STEP},
            "precision_target": config.PRECISION_TARGET,
            "recall_target": config.RECALL_TARGET,
            "k_min": config.K_MIN,
            "k_max": config.K_MAX,
        },
        "dataset": dataset,
        "models": model_results,
    }

    _sanity_checks(results)

    with open(config.RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    write_csvs(results)

    print(f"\nWrote:\n  {config.RESULTS_JSON}\n  {config.AGGREGATE_CSV}\n  {config.PER_QUERY_CSV}")
    print("Next: python src/report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
