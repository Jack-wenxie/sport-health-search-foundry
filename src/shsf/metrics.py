"""Simple IR metrics for BEIR-style qrels."""

from __future__ import annotations

import math
from typing import Mapping


def _gain(rel: int) -> float:
    return (2.0**rel) - 1.0


def ndcg_at_k(ranked_doc_ids: list[str], rels: Mapping[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = int(rels.get(doc_id, 0))
        if rel > 0:
            dcg += _gain(rel) / math.log2(i + 1)
    ideal = sorted((int(v) for v in rels.values() if int(v) > 0), reverse=True)
    idcg = sum(_gain(rel) / math.log2(i + 1) for i, rel in enumerate(ideal[:k], start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids: list[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in rels.items() if int(rel) > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def mrr_at_k(ranked_doc_ids: list[str], rels: Mapping[str, int], k: int) -> float:
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if int(rels.get(doc_id, 0)) > 0:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
