"""Cosine-similarity retrieval over a fixed corpus.

Embeddings are already L2-normalized by the Embedder, so cosine similarity is a
plain matrix product: scores[i, j] = query_i . passage_j.
"""
from __future__ import annotations

import numpy as np


class Retriever:
    def __init__(self, passage_ids: list[str], passage_embeddings: np.ndarray):
        self.passage_ids = passage_ids
        self.passage_embeddings = passage_embeddings  # (n_passages, dim), unit norm

    def score_all(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Return the full (n_queries, n_passages) cosine score matrix."""
        return query_embeddings @ self.passage_embeddings.T

    def rank(self, query_embeddings: np.ndarray) -> list[list[tuple[str, float]]]:
        """For each query, return all passages ranked by descending cosine score.

        Returns a list (per query) of (passage_id, score) tuples, fully sorted,
        so any top-k or threshold slice can be taken downstream.
        """
        scores = self.score_all(query_embeddings)
        ranked: list[list[tuple[str, float]]] = []
        for row in scores:
            order = np.argsort(-row)  # descending
            ranked.append([(self.passage_ids[j], float(row[j])) for j in order])
        return ranked
