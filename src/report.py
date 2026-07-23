"""Build the self-contained HTML report from results.json.

Renders matplotlib charts to base64-embedded PNGs (no external assets) and lays
out: executive summary + recommendation, setup, aggregate metrics, per-category
performance, score-distribution/separability, threshold calibration + dynamic-k,
per-query side-by-side deep dive, latency, and caveats.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402

PALETTE = {
    "minilm": "#2563eb",       # blue
    "e5": "#d97706",           # amber
    "granite": "#0891b2",      # cyan
    "f2llm": "#059669",        # green
    "gecko_f32": "#7c3aed",    # violet
    "gecko_quant": "#db2777",  # pink
}
# fallback colors for any model keys not in PALETTE
_FALLBACK_COLORS = ["#dc2626", "#0891b2", "#ca8a04", "#4b5563"]


def color_for(key: str, idx: int) -> str:
    return PALETTE.get(key, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
CATEGORY_ORDER = [
    "exact_paraphrase", "semantic", "partial_related", "edge_ambiguous",
    "edge_acronym", "edge_numeric", "edge_typo", "edge_lexical_trap",
    "edge_negation", "no_match",
]


def load_results() -> dict:
    with open(config.RESULTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(b64: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;height:auto;">'


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_metrics_by_k(models) -> str:
    ks = config.K_VALUES
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("recall", "ndcg"), axes):
        for i, (key, m) in enumerate(models.items()):
            ys = [m["aggregate"][f"{metric}@{k}"] for k in ks]
            ax.plot(ks, ys, marker="o", label=m["name"], color=color_for(key, i))
        ax.set_title(f"{metric.upper()}@k")
        ax.set_xlabel("k")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_category(models) -> str:
    cats = [c for c in CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    import numpy as np
    x = np.arange(len(cats))
    n = len(models)
    width = 0.8 / n
    for i, (key, m) in enumerate(models.items()):
        ys = [m["by_category"].get(c, {}).get("ndcg@10", float("nan")) for c in cats]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=m["name"], color=color_for(key, i))
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Per-category ranking quality (nDCG@10; higher = better)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_score_dist(models) -> str:
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for i, (ax, (key, m)) in enumerate(zip(axes, models.items())):
        sd = m["score_dist"]
        ax.hist(sd["irrelevant"], bins=30, alpha=0.6, label="irrelevant", color="#9ca3af", density=True)
        ax.hist(sd["relevant"], bins=30, alpha=0.7, label="relevant", color=color_for(key, i), density=True)
        tau = m["calibration"]["operating_points"]["f1_optimal"]["tau"]
        ax.axvline(tau, color="#111827", linestyle="--", linewidth=1.4, label=f"F1-opt τ={tau}")
        ax.set_title(m["name"], fontsize=10)
        ax.set_xlabel("cosine score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Score distributions: relevant vs. irrelevant (separation = threshold safety)", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_pr(models) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (key, m) in enumerate(models.items()):
        pr = m["calibration"]["pr_curve"]
        ap = m["calibration"]["average_precision"]
        ax.plot(pr["recall"], pr["precision"], color=color_for(key, i),
                label=f"{m['name']} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall (pooled query-passage decisions)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_f1_tau(models) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (key, m) in enumerate(models.items()):
        sweep = m["calibration"]["sweep"]
        taus = [r["tau"] for r in sweep]
        f1 = [r["f1"] for r in sweep]
        ax.plot(taus, f1, color=color_for(key, i), label=m["name"])
        op = m["calibration"]["operating_points"]["f1_optimal"]
        ax.scatter([op["tau"]], [op["f1"]], color=color_for(key, i), edgecolor="#111827", zorder=5, s=60)
    ax.set_xlabel("cosine threshold τ")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs. threshold (dot = F1-optimal τ per model)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v, nd=3) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
        if f != f:  # NaN
            return "–"
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return esc(v)


def recommendation(models) -> tuple[str, list[str]]:
    keys = list(models.keys())
    wins = {k: 0 for k in keys}
    scorecard = []  # (label, {key: value}, winner_key_or_None)

    metric_specs = [
        ("nDCG@10", lambda m: m["aggregate"]["ndcg@10"], True),
        ("Recall@5", lambda m: m["aggregate"]["recall@5"], True),
        ("MRR", lambda m: m["aggregate"]["mrr"], True),
        ("Score separability", lambda m: m["aggregate"]["separability"], True),
        ("Average Precision", lambda m: m["calibration"]["average_precision"], True),
        ("Dynamic-k F1", lambda m: m["calibration"]["dynamic_k"]["f1"], True),
        ("False-answer rate", lambda m: m["calibration"]["dynamic_k"]["false_answer_rate"], False),
        ("Query latency (ms)", lambda m: m["timing"]["query_per_item_ms"], False),
    ]
    for label, getter, higher in metric_specs:
        vals = {k: getter(models[k]) for k in keys}
        valid = {k: v for k, v in vals.items() if v == v}  # drop NaN
        if not valid:
            continue
        best_val = (max if higher else min)(valid.values())
        tied = [k for k, v in valid.items() if abs(v - best_val) < 1e-9]
        winner = tied[0] if len(tied) == 1 else None
        if winner:
            wins[winner] += 1
        scorecard.append((label, vals, winner))

    n_metrics = len(scorecard)
    # recommend the model with the most metric wins; tie-break on nDCG@10
    winner_key = max(keys, key=lambda k: (wins[k], models[k]["aggregate"]["ndcg@10"]))
    winner = models[winner_key]["name"]

    header = ("<tr><th>Metric</th>"
              + "".join(f"<th>{esc(models[k]['name'])}</th>" for k in keys)
              + "<th>Best</th></tr>")
    body = []
    for label, vals, win in scorecard:
        cells = "".join(f'<td class="{"win" if win == k else ""}">{fmt(vals[k])}</td>' for k in keys)
        best_txt = esc(models[win]["name"]) if win else "tie"
        body.append(f"<tr><td>{esc(label)}</td>{cells}<td>{best_txt}</td></tr>")
    wins_cells = "".join(f"<td><b>{wins[k]}</b></td>" for k in keys)
    body.append(f'<tr><td><b>Metrics won</b></td>{wins_cells}<td></td></tr>')
    table = (f'<div class="scroll"><table><thead>{header}</thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')

    notes = [
        f"<b>Recommended: {esc(winner)}</b> — best in {wins[winner_key]} of {n_metrics} "
        f"comparison metrics across the {len(keys)} models on this English-only corpus.",
        "The models cluster near-ceiling on ranking quality here, so the deciding factors are "
        "<b>score separability</b> (how safely you can set a similarity cutoff) and operational "
        "cost (latency, model size).",
        "Each model sits on its own cosine scale, so each needs its <b>own</b> calibrated "
        "threshold τ (see the calibration section) — never share a single cutoff across models.",
    ]
    return table, notes


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def aggregate_table(models) -> str:
    keys = list(models.keys())
    ks = config.K_VALUES
    header = "<tr><th>Metric</th>" + "".join(f"<th>{esc(models[k]['name'])}</th>" for k in keys) + "</tr>"
    rows = []
    metric_specs = [(f"{m}@{k}", f"{m.upper()}@{k}") for m in ("recall", "precision", "hit", "ndcg") for k in ks]
    metric_specs += [("mrr", "MRR"), ("separability", "Separability")]
    for mkey, label in metric_specs:
        vals = {k: models[k]["aggregate"][mkey] for k in keys}
        best = max(vals.values())
        cells = "".join(
            f'<td class="{"win" if abs(vals[k]-best)<1e-9 else ""}">{fmt(vals[k])}</td>' for k in keys
        )
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    return f'<div class="scroll"><table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def calibration_table(models) -> str:
    keys = list(models.keys())
    header = "<tr><th>Operating point</th>" + "".join(f"<th>{esc(models[k]['name'])}</th>" for k in keys) + "</tr>"
    rows = []
    op_labels = list(models[keys[0]]["calibration"]["operating_points"].keys())
    for op in op_labels:
        cells = []
        for k in keys:
            o = models[k]["calibration"]["operating_points"][op]
            if o is None:
                cells.append("<td>not reachable</td>")
            else:
                cells.append(f'<td>τ={fmt(o["tau"],2)} &middot; P={fmt(o["precision"])} &middot; R={fmt(o["recall"])} &middot; F1={fmt(o["f1"])}</td>')
        rows.append(f"<tr><td>{esc(op)}</td>{''.join(cells)}</tr>")
    # dynamic-k row
    cells = []
    for k in keys:
        d = models[k]["calibration"]["dynamic_k"]
        cells.append(
            f'<td>τ={fmt(d["tau"],2)} &middot; P={fmt(d["precision"])} &middot; R={fmt(d["recall"])} '
            f'&middot; F1={fmt(d["f1"])}<br>mean chunks={fmt(d["mean_chunks"],2)} '
            f'(range {d["min_chunks"]}–{d["max_chunks"]}) &middot; false-answer rate={fmt(d["false_answer_rate"])}</td>'
        )
    rows.append(f'<tr><td><b>Dynamic-k policy [{config.K_MIN}–{config.K_MAX}]</b></td>{"".join(cells)}</tr>')
    return f'<div class="scroll"><table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def per_query_section(results) -> str:
    models = results["models"]
    keys = list(models.keys())
    queries = {q["id"]: q for q in results["dataset"]["queries"]}
    ptext = {p["id"]: p["text"] for p in results["dataset"]["passages"]}

    def snippet(pid):
        return esc(ptext.get(pid, pid))[:70]

    # order: category priority, then flag disagreements first within a section
    ordered = sorted(
        queries.values(),
        key=lambda q: (CATEGORY_ORDER.index(q["category"]) if q["category"] in CATEGORY_ORDER else 99, q["id"]),
    )

    blocks = []
    for q in ordered:
        qid = q["id"]
        top1 = {k: (models[k]["per_query"][qid]["ranked_top"][0]["id"]
                    if models[k]["per_query"][qid]["ranked_top"] else None) for k in keys}
        disagree = len(set(top1.values())) > 1
        gold_ranks = {k: models[k]["per_query"][qid]["gold_rank"] for k in keys}

        # side-by-side top-5
        rows = []
        for rank in range(5):
            cells = []
            for k in keys:
                rt = models[k]["per_query"][qid]["ranked_top"]
                if rank < len(rt):
                    row = rt[rank]
                    gold = row["grade"] > 0
                    cls = "gold" if gold else ""
                    cells.append(
                        f'<td class="{cls}"><span class="pid">{esc(row["id"])}</span> '
                        f'<span class="sc">{fmt(row["score"],3)}</span><br>'
                        f'<span class="txt">{snippet(row["id"])}</span></td>'
                    )
                else:
                    cells.append("<td></td>")
            rows.append(f"<tr><td class='rk'>{rank+1}</td>{''.join(cells)}</tr>")

        gold_note = " &middot; ".join(
            f"{esc(models[k]['name'])} gold rank: {gold_ranks[k] if gold_ranks[k] else 'not in top-50'}"
            for k in keys
        ) if q["qrels"] else "no relevant passage (unanswerable)"

        badge = '<span class="badge dis">disagreement</span>' if disagree else ""
        header = "<tr><th class='rk'>#</th>" + "".join(f"<th>{esc(models[k]['name'])}</th>" for k in keys) + "</tr>"
        blocks.append(
            f'<div class="qcard">'
            f'<div class="qhead"><span class="cat">{esc(q["category"])}</span> '
            f'<span class="qtext">{esc(q["text"])}</span> {badge}</div>'
            f'<div class="gold-note">{gold_note}</div>'
            f'<div class="scroll"><table class="pq">{header}{"".join(rows)}</table></div>'
            f'</div>'
        )
    return "".join(blocks)


def latency_table(models) -> str:
    keys = list(models.keys())
    header = "<tr><th></th>" + "".join(f"<th>{esc(models[k]['name'])}</th>" for k in keys) + "</tr>"
    rows = [
        ("Params", lambda m: m["params"]),
        ("Embedding dim", lambda m: m["dim"]),
        ("Query prefix", lambda m: repr(m["query_prefix"]) if m["query_prefix"] else "(none)"),
        ("Passage prefix", lambda m: repr(m["passage_prefix"]) if m["passage_prefix"] else "(none)"),
        ("Corpus encode / item (ms)", lambda m: fmt(m["timing"]["corpus_per_item_ms"], 2)),
        ("Query encode / item (ms)", lambda m: fmt(m["timing"]["query_per_item_ms"], 2)),
    ]
    body = "".join(
        f"<tr><td>{lbl}</td>" + "".join(f"<td>{esc(getter(models[k]))}</td>" for k in keys) + "</tr>"
        for lbl, getter in rows
    )
    return f'<div class="scroll"><table><thead>{header}</thead><tbody>{body}</tbody></table></div>'


CSS = """
:root{--fg:#1f2937;--muted:#6b7280;--bd:#e5e7eb;--bg:#ffffff;--accent:#2563eb;--goldbg:#ecfdf5;--gold:#065f46;--winbg:#eff6ff;}
*{box-sizing:border-box;-webkit-print-color-adjust:exact;print-color-adjust:exact;}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);max-width:1080px;margin:0 auto;padding:28px 20px 80px;line-height:1.5;}
h1{font-size:1.7rem;margin:0 0 4px;} h2{font-size:1.25rem;margin:34px 0 10px;border-bottom:2px solid var(--bd);padding-bottom:6px;}
h3{font-size:1.02rem;margin:18px 0 6px;}
.sub{color:var(--muted);margin:0 0 18px;}
table{border-collapse:collapse;width:100%;font-size:0.86rem;margin:8px 0;}
th,td{border:1px solid var(--bd);padding:6px 9px;text-align:left;vertical-align:top;}
th{background:#f9fafb;font-weight:600;}
td.win{background:var(--winbg);font-weight:600;}
.scroll{overflow-x:auto;}
.callout{background:#f9fafb;border:1px solid var(--bd);border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px;margin:12px 0;}
.callout ul{margin:6px 0;padding-left:20px;} .callout li{margin:4px 0;}
.qcard{border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin:12px 0;}
.qhead{font-size:0.95rem;margin-bottom:4px;}
.cat{display:inline-block;background:#eef2ff;color:#3730a3;font-size:0.72rem;padding:2px 8px;border-radius:10px;margin-right:6px;font-weight:600;}
.qtext{font-weight:600;}
.gold-note{color:var(--muted);font-size:0.8rem;margin-bottom:8px;}
.badge.dis{background:#fef2f2;color:#991b1b;font-size:0.72rem;padding:2px 8px;border-radius:10px;font-weight:600;}
table.pq td.gold{background:var(--goldbg);}
table.pq td.gold .pid{color:var(--gold);font-weight:700;}
.pid{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.78rem;}
.sc{color:var(--muted);font-size:0.78rem;float:right;}
.txt{color:#374151;font-size:0.78rem;}
td.rk,th.rk{width:26px;text-align:center;color:var(--muted);}
.charts{margin:10px 0;} .two{display:flex;gap:16px;flex-wrap:wrap;} .two>div{flex:1;min-width:320px;}
footer{margin-top:40px;color:var(--muted);font-size:0.8rem;}
code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:0.85em;}
@media print{
  body{max-width:none;padding:0;font-size:0.8rem;}
  h2{page-break-after:avoid;} h1,h2,h3{break-after:avoid;}
  .qcard,.callout,table,.charts,.two{break-inside:avoid;page-break-inside:avoid;}
  a{color:inherit;text-decoration:none;}
}
@page{size:A4;margin:14mm 12mm;}
"""


def build_html(results) -> str:
    models = results["models"]
    ds = results["dataset"]
    rec_table, rec_notes = recommendation(models)

    c_k = chart_metrics_by_k(models)
    c_cat = chart_category(models)
    c_dist = chart_score_dist(models)
    c_pr = chart_pr(models)
    c_f1 = chart_f1_tau(models)

    ncats = len({q["category"] for q in ds["queries"]})
    rec_notes_html = "".join(f"<li>{n}</li>" for n in rec_notes)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Embedding Model Comparison Report</title><style>{CSS}</style></head><body>
<h1>Embedding Model Comparison &mdash; Top-k Retrieval</h1>
<p class="sub">{esc(" vs. ".join(m["name"] for m in models.values()))} &middot;
{ds['meta']['n_passages']} passages, {ds['meta']['n_queries']} queries across {ncats} categories &middot;
English-only, mixed general knowledge &middot; generated {esc(results['generated_at'][:19])} UTC</p>

<h2>1. Executive summary &amp; recommendation</h2>
<div class="callout"><ul>{rec_notes_html}</ul></div>
{rec_table}

<h2>2. Setup</h2>
<p class="sub">Each model uses its own prefix convention and L2-normalized cosine similarity.
Same corpus, queries, qrels, and k grid for every model. Run on CPU.</p>
{latency_table(models)}
<div class="callout"><b>Fairness note.</b> <code>multilingual-e5-small</code> requires
<code>query:</code> / <code>passage:</code> prefixes; <code>all-MiniLM-L6-v2</code> uses none.
Both are applied automatically. The corpus is <b>English-only</b>, so e5's multilingual strength
is <i>not</i> exercised here &mdash; this is a pure English semantic-retrieval comparison.
<br><br>The two <code>Gecko-110m-en</code> models are <b>on-device LiteRT/TFLite</b> embedders
(<code>f32</code> and dynamic-<code>int8</code> variants of the same model) run through a TFLite
interpreter with a SentencePiece tokenizer, whereas the other three run under batched PyTorch.
<b>Latency is therefore not a fair head-to-head:</b> the Gecko numbers are single-item (batch&nbsp;1),
pay a fixed 512-token padding cost per text, and include SentencePiece tokenization &mdash; read them
as on-device edge latency, not throughput against the server models. Retrieval-<i>quality</i> metrics
remain directly comparable.</div>

<h2>3. Aggregate metrics</h2>
<p class="sub">Averaged over answerable queries; best value per row highlighted. k grid: {config.K_VALUES}.</p>
{aggregate_table(models)}
<div class="charts">{img_tag(c_k, "metrics by k")}</div>

<h2>4. Per-category performance</h2>
<p class="sub">Where each model wins or loses, especially on the adversarial edge cases.</p>
<div class="charts">{img_tag(c_cat, "per-category nDCG@10")}</div>

<h2>5. Score-distribution &amp; separability</h2>
<p class="sub">How well each model separates relevant from irrelevant chunks. A wide gap means a
similarity cutoff is safe; overlapping distributions mean thresholding is risky. The dashed line
is each model's F1-optimal &tau;.</p>
<div class="charts">{img_tag(c_dist, "score distributions")}</div>

<h2>6. Threshold calibration &amp; dynamic-k policy</h2>
<p class="sub">Per-model cosine cutoffs (they differ because the models sit on different score
scales) and the deployable retrieval rule: keep chunks with score &ge; &tau;, clamped to
[{config.K_MIN}, {config.K_MAX}] chunks.</p>
<div class="two"><div>{img_tag(c_pr, "PR curves")}</div><div>{img_tag(c_f1, "F1 vs tau")}</div></div>
{calibration_table(models)}
<div class="callout">The <b>dynamic-k precision</b> is bounded by the <code>k_min={config.K_MIN}</code>
floor: when a query has only 1&ndash;2 relevant passages, returning at least {config.K_MIN} chunks
necessarily includes irrelevant ones. The floor is a deliberate trade to avoid starving the LLM
context; recall and the false-answer rate are the more meaningful numbers for this policy.</div>

<h2>7. Per-query side-by-side deep dive</h2>
<p class="sub">Top-5 retrieved chunks per model with cosine scores. <span class="gold" style="padding:1px 6px;border-radius:4px;">Green</span>
= a relevant (gold) passage. Ordered by category; queries where the models disagree on the #1 result are badged.</p>
{per_query_section(results)}

<h2>8. Caveats</h2>
<div class="callout"><ul>
<li><b>Synthetic data.</b> A hand-authored 50-passage corpus is small; absolute metrics are optimistic and category-level numbers rest on a few queries each. Use the <i>relative</i> comparison, not absolute scores.</li>
<li><b>CPU timing.</b> Latencies are CPU-only and indicative, not production throughput.</li>
<li><b>English-only.</b> e5's multilingual capability is untested here; if your real workload is multilingual, re-run with a multilingual corpus before deciding.</li>
<li><b>Swap in real data.</b> Replace <code>data/dataset.json</code> with your own passages/queries/qrels and re-run <code>evaluate.py</code> + <code>report.py</code> unchanged.</li>
</ul></div>

<footer>Generated by the embedding comparison harness &middot; raw data in results.json / aggregate.csv / per_query.csv</footer>
</body></html>"""
    return html


def main() -> int:
    results = load_results()
    html = build_html(results)
    with open(config.REPORT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {config.REPORT_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
