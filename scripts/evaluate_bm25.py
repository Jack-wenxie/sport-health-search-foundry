from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shsf.beir_io import load_beir_corpus, load_beir_qrels, load_beir_queries
from shsf.bm25 import BM25Index
from shsf.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "nfcorpus")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "raw" / "nfcorpus-bm25-baseline.json")
    args = parser.parse_args()

    corpus_path = args.dataset_dir / "corpus.jsonl"
    queries_path = args.dataset_dir / "queries.jsonl"
    qrels_path = args.dataset_dir / "qrels" / f"{args.split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise SystemExit(f"Missing NFCorpus file: {path}")

    docs = load_beir_corpus(corpus_path)
    queries = load_beir_queries(queries_path)
    qrels = load_beir_qrels(qrels_path)
    index = BM25Index.build(docs)

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    evaluated = 0
    for qid in sorted(qrels):
        query = queries.get(qid)
        if not query:
            continue
        ranked = [r.doc_id for r in index.search(query, top_k=max(args.top_k, 100))]
        rels = qrels[qid]
        ndcgs.append(ndcg_at_k(ranked, rels, args.top_k))
        recalls.append(recall_at_k(ranked, rels, args.top_k))
        mrrs.append(mrr_at_k(ranked, rels, args.top_k))
        evaluated += 1

    result = {
        "run_id": "nfcorpus-local-bm25",
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "BEIR NFCorpus",
        "split": args.split,
        "index": "local dependency-free BM25",
        "analyzer_note": "Regex tokenization; not guaranteed to match BEIR official Anserini/Pyserini analyzer exactly.",
        "documents": len(docs),
        "queries_with_qrels": len(qrels),
        "queries_evaluated": evaluated,
        "metrics": {
            f"ndcg@{args.top_k}": round(mean(ndcgs), 6),
            f"recall@{args.top_k}": round(mean(recalls), 6),
            f"mrr@{args.top_k}": round(mean(mrrs), 6),
        },
        "official_baseline_note": "Compare against BEIR official BM25/NFCorpus baseline in the final evidence report; exact parity may require official analyzer stack.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
