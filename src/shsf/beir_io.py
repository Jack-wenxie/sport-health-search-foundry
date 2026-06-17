"""BEIR-format dataset loaders."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_beir_corpus(path: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                {
                    "doc_id": str(row.get("_id") or row.get("doc_id")),
                    "title": str(row.get("title", "")),
                    "abstract": str(row.get("text", "")),
                }
            )
    docs.sort(key=lambda x: x["doc_id"])
    return docs


def load_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("_id") or row.get("query_id"))
            queries[qid] = str(row.get("text", ""))
    return queries


def load_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if rows and rows[0] and rows[0][0].lower() in {"query-id", "query_id", "qid"}:
        rows = rows[1:]
    for row in rows:
        if len(row) < 3:
            continue
        qid, doc_id, score = row[0], row[1], row[2]
        try:
            rel = int(float(score))
        except ValueError:
            continue
        qrels.setdefault(str(qid), {})[str(doc_id)] = rel
    return qrels
