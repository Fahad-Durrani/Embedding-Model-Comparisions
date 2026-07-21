"""Retrieval metrics computed from ranked (passage_id, score) lists + qrels.

Binary metrics (recall/precision/hit/MRR) use a relevance-grade cutoff; nDCG
uses the raw graded judgments. Queries with an empty relevant set (no_match and
the unanswerable negation queries) are *excluded* from ranking-quality averages
— those are handled by the threshold/false-positive diagnostics instead.
"""
from __future__ import annotations

import math
from collections import defaultdict

Ranked = list[tuple[str, float]]      # sorted desc by score
Qrels = dict[str, int]                # passage_id -> grade


def relevant_set(qrels: Qrels, grade_threshold: int) -> set[str]:
    return {pid for pid, g in qrels.items() if g >= grade_threshold}


def recall_at_k(ranked: Ranked, rel: set[str], k: int) -> float:
    if not rel:
        return float("nan")
    topk = {pid for pid, _ in ranked[:k]}
    return len(topk & rel) / len(rel)


def precision_at_k(ranked: Ranked, rel: set[str], k: int) -> float:
    if k == 0:
        return float("nan")
    topk = [pid for pid, _ in ranked[:k]]
    hits = sum(1 for pid in topk if pid in rel)
    return hits / k


def hit_at_k(ranked: Ranked, rel: set[str], k: int) -> float:
    if not rel:
        return float("nan")
    topk = {pid for pid, _ in ranked[:k]}
    return 1.0 if topk & rel else 0.0


def reciprocal_rank(ranked: Ranked, rel: set[str]) -> float:
    if not rel:
        return float("nan")
    for i, (pid, _) in enumerate(ranked, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Ranked, qrels: Qrels, k: int) -> float:
    if not qrels:
        return float("nan")
    dcg = 0.0
    for i, (pid, _) in enumerate(ranked[:k], start=1):
        gain = qrels.get(pid, 0)
        if gain:
            dcg += gain / math.log2(i + 1)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else float("nan")


def separability(ranked: Ranked, rel: set[str]) -> float:
    """max relevant score - max irrelevant score for one query (higher = better)."""
    if not rel:
        return float("nan")
    rel_scores = [s for pid, s in ranked if pid in rel]
    irr_scores = [s for pid, s in ranked if pid not in rel]
    if not rel_scores or not irr_scores:
        return float("nan")
    return max(rel_scores) - max(irr_scores)


def _nanmean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def per_query_metrics(
    ranked: Ranked, qrels: Qrels, k_values: list[int], grade_threshold: int
) -> dict:
    rel = relevant_set(qrels, grade_threshold)
    out: dict = {"answerable": bool(rel)}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        out[f"precision@{k}"] = precision_at_k(ranked, rel, k)
        out[f"hit@{k}"] = hit_at_k(ranked, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels, k)
    out["mrr"] = reciprocal_rank(ranked, rel)
    out["separability"] = separability(ranked, rel)
    out["top_score"] = ranked[0][1] if ranked else float("nan")
    # rank of the first relevant passage (None if never retrieved / unanswerable)
    out["gold_rank"] = None
    if rel:
        for i, (pid, _) in enumerate(ranked, start=1):
            if pid in rel:
                out["gold_rank"] = i
                break
    return out


def aggregate(per_query: dict[str, dict], k_values: list[int]) -> dict:
    """Mean of each metric across answerable queries (NaNs ignored)."""
    keys = [f"{m}@{k}" for m in ("recall", "precision", "hit", "ndcg") for k in k_values]
    keys += ["mrr", "separability"]
    agg = {}
    for key in keys:
        agg[key] = _nanmean([pq[key] for pq in per_query.values()])
    return agg


def aggregate_by_category(
    per_query: dict[str, dict], categories: dict[str, str], k_values: list[int]
) -> dict[str, dict]:
    buckets: dict[str, dict[str, dict]] = defaultdict(dict)
    for qid, pq in per_query.items():
        buckets[categories[qid]][qid] = pq
    return {cat: aggregate(pqs, k_values) for cat, pqs in buckets.items()}
