"""Per-model threshold calibration and the dynamic-k retrieval policy.

The two models sit on different cosine scales, so each is calibrated
independently. We pool every (query, passage) decision, label it from qrels,
and sweep the cosine cutoff tau to trace precision / recall / F1. From that we
pick operating points and then *simulate* the deployable retrieval rule:

    keep chunks with score >= tau, clamp the count into [k_min, k_max].

Unanswerable queries (no_match + unanswerable negations) contribute only
negatives, so a model that hands out high scores to junk is correctly punished.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from metrics import relevant_set


def pooled_pairs(ranked_by_query, qrels_by_query, grade_threshold):
    """Flatten to (y_true, y_score) over every (query, passage) pair."""
    y_true, y_score = [], []
    for qid, ranked in ranked_by_query.items():
        rel = relevant_set(qrels_by_query[qid], grade_threshold)
        for pid, score in ranked:
            y_true.append(1 if pid in rel else 0)
            y_score.append(score)
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def sweep(y_true, y_score, taus):
    total_rel = int(y_true.sum())
    rows = []
    for tau in taus:
        pred = y_score >= tau
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        n_pred = tp + fp
        precision = tp / n_pred if n_pred > 0 else 1.0
        recall = tp / total_rel if total_rel > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {"tau": round(float(tau), 4), "precision": precision,
             "recall": recall, "f1": f1, "tp": tp, "fp": fp, "n_pred": n_pred}
        )
    return rows


def _operating_points(rows, precision_target, recall_target):
    f1_opt = max(rows, key=lambda r: (r["f1"], r["recall"]))

    hi_prec = None
    for r in rows:  # ascending tau; first (smallest tau) meeting precision target
        if r["precision"] >= precision_target and r["n_pred"] > 0:
            hi_prec = r
            break

    hi_rec = None
    for r in rows:  # ascending tau; keep the last (largest tau) still meeting recall
        if not np.isnan(r["recall"]) and r["recall"] >= recall_target:
            hi_rec = r

    def pack(r):
        if r is None:
            return None
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}

    return {
        "f1_optimal": pack(f1_opt),
        f"high_precision@{precision_target:g}": pack(hi_prec),
        f"high_recall@{recall_target:g}": pack(hi_rec),
    }


def simulate_dynamic_k(ranked_by_query, qrels_by_query, grade_threshold, tau, k_min, k_max):
    """Apply the keep>=tau, clamp-to-[k_min,k_max] policy and measure it."""
    tp = fp = fn = 0
    counts = []
    per_cat_counts: dict[str, list[int]] = {}
    unanswerable_total = 0
    unanswerable_flagged_answer = 0  # top score >= tau despite no relevant passage
    per_query = {}

    for qid, ranked in ranked_by_query.items():
        rel = relevant_set(qrels_by_query[qid], grade_threshold)
        above = [(pid, s) for pid, s in ranked if s >= tau]
        n = min(max(len(above), k_min), k_max)
        selected = ranked[:n]
        sel_ids = {pid for pid, _ in selected}

        q_tp = len(sel_ids & rel)
        q_fp = len(sel_ids - rel)
        q_fn = len(rel - sel_ids)
        tp += q_tp
        fp += q_fp
        fn += q_fn
        counts.append(n)

        top_score = ranked[0][1] if ranked else float("-inf")
        if not rel:
            unanswerable_total += 1
            if top_score >= tau:
                unanswerable_flagged_answer += 1

        per_query[qid] = {
            "n_returned": n,
            "n_above_tau": len(above),
            "top_score": round(top_score, 4),
            "flagged_no_answer": bool(top_score < tau),
            "selected": [{"id": pid, "score": round(s, 4)} for pid, s in selected],
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else 0.0

    return {
        "tau": round(float(tau), 4),
        "k_min": k_min,
        "k_max": k_max,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_chunks": round(float(np.mean(counts)), 3),
        "median_chunks": int(np.median(counts)),
        "min_chunks": int(np.min(counts)),
        "max_chunks": int(np.max(counts)),
        "unanswerable_total": unanswerable_total,
        "unanswerable_false_answer": unanswerable_flagged_answer,
        "false_answer_rate": round(unanswerable_flagged_answer / unanswerable_total, 4) if unanswerable_total else float("nan"),
        "per_query": per_query,
    }


def calibrate_model(
    ranked_by_query, qrels_by_query, grade_threshold,
    tau_min, tau_max, tau_step, precision_target, recall_target, k_min, k_max,
):
    y_true, y_score = pooled_pairs(ranked_by_query, qrels_by_query, grade_threshold)
    taus = np.round(np.arange(tau_min, tau_max + 1e-9, tau_step), 4)
    rows = sweep(y_true, y_score, taus)
    ops = _operating_points(rows, precision_target, recall_target)

    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    op_tau = ops["f1_optimal"]["tau"]
    dyn = simulate_dynamic_k(
        ranked_by_query, qrels_by_query, grade_threshold, op_tau, k_min, k_max
    )

    return {
        "sweep": rows,
        "operating_points": ops,
        "average_precision": round(ap, 4),
        "pr_curve": {"precision": [round(float(p), 4) for p in prec],
                     "recall": [round(float(r), 4) for r in rec]},
        "dynamic_k": dyn,
    }
