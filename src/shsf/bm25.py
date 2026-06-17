"""Small dependency-free BM25 implementation for deterministic smoke tests."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int


class BM25Index:
    def __init__(
        self,
        doc_ids: list[str],
        doc_lengths: list[int],
        term_freqs: list[dict[str, int]],
        doc_freqs: dict[str, int],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.doc_ids = doc_ids
        self.doc_lengths = doc_lengths
        self.term_freqs = term_freqs
        self.doc_freqs = doc_freqs
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    @classmethod
    def build(cls, docs: Iterable[dict[str, str]]) -> "BM25Index":
        doc_ids: list[str] = []
        doc_lengths: list[int] = []
        term_freqs: list[dict[str, int]] = []
        doc_freqs: Counter[str] = Counter()

        for doc in docs:
            doc_id = str(doc["doc_id"])
            text = " ".join(str(doc.get(k, "")) for k in ("title", "abstract"))
            tokens = tokenize(text)
            counts = Counter(tokens)
            doc_ids.append(doc_id)
            doc_lengths.append(len(tokens))
            term_freqs.append(dict(counts))
            doc_freqs.update(counts.keys())

        return cls(doc_ids, doc_lengths, term_freqs, dict(sorted(doc_freqs.items())))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms or not self.doc_ids:
            return []

        scores: list[tuple[str, float]] = []
        n_docs = len(self.doc_ids)
        avg_len = self.avg_doc_len or 1.0

        for idx, doc_id in enumerate(self.doc_ids):
            score = 0.0
            doc_len = self.doc_lengths[idx] or 1
            tf_map = self.term_freqs[idx]
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avg_len)
                score += idf * (tf * (self.k1 + 1.0) / denom)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(doc_id=doc_id, score=round(score, 8), rank=rank)
            for rank, (doc_id, score) in enumerate(scores[:top_k], start=1)
        ]

    def to_dict(self) -> dict:
        return {
            "index_type": "bm25",
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "term_freqs": self.term_freqs,
            "doc_freqs": self.doc_freqs,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BM25Index":
        return cls(
            doc_ids=list(payload["doc_ids"]),
            doc_lengths=[int(x) for x in payload["doc_lengths"]],
            term_freqs=[{str(k): int(v) for k, v in d.items()} for d in payload["term_freqs"]],
            doc_freqs={str(k): int(v) for k, v in payload["doc_freqs"].items()},
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
