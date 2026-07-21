# Embedding Model Comparison Harness

Side-by-side benchmark of two sentence-embedding models for a **top-k chunk
retrieval** pipeline, with per-query scores, threshold calibration, a deployable
dynamic-k policy, and a detailed HTML report.

Models under test:

| key | model | prefix convention | dim | params |
|-----|-------|-------------------|-----|--------|
| `minilm` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | none | 384 | ~22M |
| `e5` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | `query:` / `passage:` | 384 | ~118M |
| `f2llm` | [`codefuse-ai/F2LLM-v2-80M`](https://huggingface.co/codefuse-ai/F2LLM-v2-80M) | `Instruct: …\nQuery: ` on query only | 320 | ~80M |

> **Fairness matters.** Each model has its own convention: e5 *requires*
> `query:` / `passage:` prefixes; F2LLM *requires* an `Instruct: …\nQuery: `
> instruction on the query (documents unprefixed); MiniLM uses none. The harness
> applies each model's own convention automatically — skipping these prefixes is
> the most common way people accidentally cripple a model.

## Quick start

```bash
pip install -r requirements.txt
python data/build_dataset.py     # -> data/dataset.json (50 passages, 40 queries)
python src/evaluate.py           # downloads models on first run; -> results/*.json|csv
python src/report.py             # -> results/report.html   (open this)
```

Then open `results/report.html` in a browser.

## What it measures

- **Ranking quality** per model, per k in `[1,3,5,10,20,40]`, overall and per
  category: Recall@k, Precision@k, Hit@k, MRR, nDCG@k.
- **Score separability** — the gap between relevant and irrelevant cosine scores;
  how safely you can set a similarity cutoff.
- **Threshold calibration** — per-model cosine cutoff sweep, PR curve + Average
  Precision, and recommended operating points (F1-optimal, high-precision,
  high-recall). *Each model gets its own τ* because they sit on different score
  scales.
- **Dynamic-k retrieval policy** — keep chunks with `score ≥ τ`, clamp the count
  to `[K_MIN, K_MAX] = [5, 40]`; reported with realized precision/recall/F1, mean
  chunks returned, and a false-answer rate on unanswerable queries.

## Dataset design

Hand-authored, deterministic English corpus (mixed general knowledge) in
`data/build_dataset.py`. Queries are tagged into 10 categories including
adversarial edge cases: `edge_negation`, `edge_lexical_trap` (e.g. "river bank"
vs "financial bank", "Python language" vs "Python snake"), `edge_numeric`,
`edge_typo`, `edge_acronym`, `edge_ambiguous`, plus `no_match` (unanswerable)
queries that test false-positive control.

Relevance judgments (`qrels`) are graded: **2** = highly relevant, **1** =
partially relevant. `no_match` and unanswerable negation queries have no
relevant passage and are excluded from ranking averages (used for the
threshold/false-positive diagnostics instead).

## Using your own data

Replace `data/dataset.json` with your real passages, queries, and qrels (same
schema), then re-run `evaluate.py` and `report.py` unchanged. Schema:

```json
{
  "passages": [{"id": "p01", "topic": "...", "text": "..."}],
  "queries":  [{"id": "q01", "category": "...", "text": "...",
                "qrels": {"p01": 2, "p07": 1}}]
}
```

## Configuration

All knobs live in `config.py`: models + prefixes, k grid, threshold sweep range,
precision/recall targets for operating points, the `[K_MIN, K_MAX]` band, and the
relevance-grade cutoff for binary metrics.

## Files

```
config.py               # all configuration
data/build_dataset.py   # deterministic synthetic dataset generator
data/dataset.json        # corpus + queries + qrels (generated)
src/embedder.py         # model wrapper: prefixes, normalize, timing
src/retriever.py        # cosine top-k retrieval
src/metrics.py          # recall/precision/hit/MRR/nDCG/separability
src/calibrate.py        # threshold sweep, PR curve, dynamic-k simulation
src/evaluate.py         # MAIN runner -> results/results.json + CSVs
src/report.py           # HTML report + charts
results/                # generated: results.json, aggregate.csv, per_query.csv, report.html
```

## Caveats

- Synthetic 50-passage corpus: absolute metrics are optimistic; trust the
  *relative* comparison. Swap in real data for production decisions.
- CPU-only latency figures are indicative, not throughput benchmarks.
- The corpus is **English-only**, so e5's multilingual capability is not tested
  here. Re-run with a multilingual corpus if that matches your workload.
