"""Model wrapper that applies each model's own prefix convention and returns
L2-normalized embeddings, with encode-timing captured for the latency report.

The prefix handling is the crux of a *fair* comparison:
  * all-MiniLM-L6-v2  -> no prefix
  * multilingual-e5-small -> "query: " on queries, "passage: " on passages
Skipping the e5 prefixes silently degrades its quality, so we never let the
caller forget them: the wrapper takes them from the model config.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EncodeTiming:
    corpus_seconds: float = 0.0
    corpus_items: int = 0
    query_seconds: float = 0.0
    query_items: int = 0

    @property
    def corpus_per_item_ms(self) -> float:
        return 1000.0 * self.corpus_seconds / self.corpus_items if self.corpus_items else 0.0

    @property
    def query_per_item_ms(self) -> float:
        return 1000.0 * self.query_seconds / self.query_items if self.query_items else 0.0

    def as_dict(self) -> dict:
        return {
            "corpus_seconds": round(self.corpus_seconds, 4),
            "corpus_items": self.corpus_items,
            "corpus_per_item_ms": round(self.corpus_per_item_ms, 3),
            "query_seconds": round(self.query_seconds, 4),
            "query_items": self.query_items,
            "query_per_item_ms": round(self.query_per_item_ms, 3),
        }


@dataclass
class Embedder:
    """Wraps a SentenceTransformer with fixed query/passage prefixes."""

    cfg: dict
    _model: SentenceTransformer | None = field(default=None, repr=False)
    timing: EncodeTiming = field(default_factory=EncodeTiming)

    @property
    def key(self) -> str:
        return self.cfg["key"]

    @property
    def name(self) -> str:
        return self.cfg["name"]

    def load(self) -> "Embedder":
        if self._model is None:
            self._model = SentenceTransformer(self.cfg["hf_id"])
        return self

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        prefixed = [f"{prefix}{t}" for t in texts]
        # normalize_embeddings=True => unit vectors, so a dot product is cosine.
        return self._model.encode(
            prefixed,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        start = time.perf_counter()
        emb = self._encode(texts, self.cfg["passage_prefix"])
        self.timing.corpus_seconds += time.perf_counter() - start
        self.timing.corpus_items += len(texts)
        return emb

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        start = time.perf_counter()
        emb = self._encode(texts, self.cfg["query_prefix"])
        self.timing.query_seconds += time.perf_counter() - start
        self.timing.query_items += len(texts)
        return emb
