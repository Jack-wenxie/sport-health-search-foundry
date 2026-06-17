"""Local dense retrieval via TF-IDF + truncated SVD (LSA).

This is a serverless dense baseline for Phase 1. It is intentionally small and
dependency-light: numpy + scipy only, no model download and no search service.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds

from .bm25 import SearchResult, tokenize


@dataclass
class DenseSearchResult:
    doc_id: str
    score: float
    rank: int


class LsaIndex:
    def __init__(
        self,
        doc_ids: list[str],
        vocab: dict[str, int],
        idf: np.ndarray,
        components: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> None:
        self.doc_ids = doc_ids
        self.vocab = vocab
        self.idf = idf.astype(np.float32)
        self.components = components.astype(np.float32)
        self.doc_embeddings = _l2_normalize(doc_embeddings.astype(np.float32))

    @classmethod
    def build(
        cls,
        docs: Iterable[dict[str, str]],
        dims: int = 128,
        max_features: int = 20000,
        min_df: int = 2,
        random_state: int = 13,
    ) -> "LsaIndex":
        doc_rows = list(docs)
        doc_ids = [str(doc["doc_id"]) for doc in doc_rows]
        token_counts: list[Counter[str]] = []
        dfs: Counter[str] = Counter()
        for doc in doc_rows:
            text = " ".join(str(doc.get(k, "")) for k in ("title", "abstract"))
            counts = Counter(tokenize(text))
            token_counts.append(counts)
            dfs.update(counts.keys())

        terms = [
            term
            for term, df in sorted(dfs.items(), key=lambda item: (-item[1], item[0]))
            if df >= min_df
        ][:max_features]
        vocab = {term: idx for idx, term in enumerate(sorted(terms))}
        if not vocab:
            raise ValueError("No vocabulary terms after min_df/max_features filtering")

        matrix = _counts_to_matrix(token_counts, vocab)
        n_docs = matrix.shape[0]
        df_vec = np.asarray((matrix > 0).sum(axis=0)).ravel()
        idf = np.log((1.0 + n_docs) / (1.0 + df_vec)) + 1.0
        tfidf = matrix @ diags(idf)

        k = min(dims, min(tfidf.shape) - 1)
        if k < 2:
            raise ValueError(f"Not enough matrix rank for dims={dims}: shape={tfidf.shape}")
        u, s, vt = svds(tfidf.astype(np.float32), k=k, random_state=random_state)
        order = np.argsort(s)[::-1]
        s = s[order]
        vt = vt[order, :]
        u = u[:, order]
        doc_embeddings = u * s
        return cls(doc_ids, vocab, idf, vt, doc_embeddings)

    def search(self, query: str, top_k: int = 10) -> list[DenseSearchResult]:
        q = self.query_embedding(query)
        if q is None:
            return []
        scores = self.doc_embeddings @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [
            DenseSearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=round(float(scores[int(idx)]), 8),
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
            if float(scores[int(idx)]) > 0
        ]

    def query_embedding(self, query: str) -> np.ndarray | None:
        counts = Counter(tokenize(query))
        cols: list[int] = []
        vals: list[float] = []
        for term, tf in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                cols.append(idx)
                vals.append(1.0 + math.log(float(tf)))
        if not cols:
            return None
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        vec[cols] = np.asarray(vals, dtype=np.float32)
        vec *= self.idf
        emb = vec @ self.components.T
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return (emb / norm).astype(np.float32)


def _counts_to_matrix(token_counts: list[Counter[str]], vocab: dict[str, int]) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx, counts in enumerate(token_counts):
        for term, tf in counts.items():
            col_idx = vocab.get(term)
            if col_idx is not None and tf > 0:
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(1.0 + math.log(float(tf)))
    return csr_matrix((data, (rows, cols)), shape=(len(token_counts), len(vocab)), dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    dense_results: list[DenseSearchResult],
    k: int = 60,
    top_k: int = 100,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    for r in bm25_results:
        scores[r.doc_id] = scores.get(r.doc_id, 0.0) + 1.0 / (k + r.rank)
    for r in dense_results:
        scores[r.doc_id] = scores.get(r.doc_id, 0.0) + 1.0 / (k + r.rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        SearchResult(doc_id=doc_id, score=round(float(score), 8), rank=rank)
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    ]


def weighted_fusion(
    bm25_results: list[SearchResult],
    dense_results: list[DenseSearchResult],
    dense_weight: float,
    top_k: int = 100,
) -> list[SearchResult]:
    bm25_scores = {r.doc_id: r.score for r in bm25_results}
    dense_scores = {r.doc_id: r.score for r in dense_results}
    norm_bm25 = _minmax(bm25_scores)
    norm_dense = _minmax(dense_scores)
    candidates = set(norm_bm25) | set(norm_dense)
    scores = {
        doc_id: (1.0 - dense_weight) * norm_bm25.get(doc_id, 0.0)
        + dense_weight * norm_dense.get(doc_id, 0.0)
        for doc_id in candidates
    }
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        SearchResult(doc_id=doc_id, score=round(float(score), 8), rank=rank)
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    ]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
