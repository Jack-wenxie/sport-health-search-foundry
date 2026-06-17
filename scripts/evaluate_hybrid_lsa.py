from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shsf.beir_io import load_beir_corpus, load_beir_qrels, load_beir_queries
from shsf.bm25 import BM25Index, SearchResult
from shsf.lsa import LsaIndex, reciprocal_rank_fusion, weighted_fusion
from shsf.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict[str, float]:
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    for qid, rels in qrels.items():
        ranked = rankings.get(qid, [])
        ndcgs.append(ndcg_at_k(ranked, rels, k))
        recalls.append(recall_at_k(ranked, rels, k))
        mrrs.append(mrr_at_k(ranked, rels, k))
    return {
        f"ndcg@{k}": round(mean(ndcgs), 6),
        f"recall@{k}": round(mean(recalls), 6),
        f"mrr@{k}": round(mean(mrrs), 6),
    }


def to_ids(results: list[SearchResult], top_k: int) -> list[str]:
    return [r.doc_id for r in results[:top_k]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "nfcorpus")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "raw" / "nfcorpus-hybrid-lsa-sweep.json")
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
    eval_qids = [qid for qid in sorted(qrels) if qid in queries]

    bm25 = BM25Index.build(docs)
    bm25_full: dict[str, list[SearchResult]] = {
        qid: bm25.search(queries[qid], top_k=args.candidate_k) for qid in eval_qids
    }
    bm25_rankings = {qid: to_ids(results, args.metric_k) for qid, results in bm25_full.items()}
    bm25_metrics = evaluate_rankings(bm25_rankings, qrels, args.metric_k)

    runs: list[dict] = [
        {
            "run_id": "bm25",
            "kind": "lexical",
            "metrics": bm25_metrics,
        }
    ]
    best = runs[0]

    for dims in args.dims:
        lsa = LsaIndex.build(docs, dims=dims, max_features=args.max_features)
        dense_full = {qid: lsa.search(queries[qid], top_k=args.candidate_k) for qid in eval_qids}
        dense_rankings = {qid: [r.doc_id for r in results[: args.metric_k]] for qid, results in dense_full.items()}
        dense_metrics = evaluate_rankings(dense_rankings, qrels, args.metric_k)
        dense_run = {
            "run_id": f"lsa-dense-d{dims}",
            "kind": "dense_lsa",
            "dims": dims,
            "metrics": dense_metrics,
        }
        runs.append(dense_run)
        if dense_metrics[f"ndcg@{args.metric_k}"] > best["metrics"][f"ndcg@{args.metric_k}"]:
            best = dense_run

        rrf_rankings = {
            qid: [r.doc_id for r in reciprocal_rank_fusion(bm25_full[qid], dense_full[qid], top_k=args.metric_k)]
            for qid in eval_qids
        }
        rrf_metrics = evaluate_rankings(rrf_rankings, qrels, args.metric_k)
        rrf_run = {
            "run_id": f"rrf-bm25-lsa-d{dims}",
            "kind": "hybrid_rrf",
            "dims": dims,
            "metrics": rrf_metrics,
        }
        runs.append(rrf_run)
        if rrf_metrics[f"ndcg@{args.metric_k}"] > best["metrics"][f"ndcg@{args.metric_k}"]:
            best = rrf_run

        for weight in args.weights:
            weighted_rankings = {
                qid: [
                    r.doc_id
                    for r in weighted_fusion(
                        bm25_full[qid],
                        dense_full[qid],
                        dense_weight=weight,
                        top_k=args.metric_k,
                    )
                ]
                for qid in eval_qids
            }
            metrics = evaluate_rankings(weighted_rankings, qrels, args.metric_k)
            run = {
                "run_id": f"weighted-bm25-lsa-d{dims}-w{weight:g}",
                "kind": "hybrid_weighted",
                "dims": dims,
                "dense_weight": weight,
                "metrics": metrics,
            }
            runs.append(run)
            if metrics[f"ndcg@{args.metric_k}"] > best["metrics"][f"ndcg@{args.metric_k}"]:
                best = run

    result = {
        "run_id": "nfcorpus-lsa-hybrid-sweep",
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "BEIR NFCorpus",
        "split": args.split,
        "documents": len(docs),
        "queries_evaluated": len(eval_qids),
        "metric_k": args.metric_k,
        "candidate_k": args.candidate_k,
        "max_features": args.max_features,
        "baseline": bm25_metrics,
        "best": best,
        "beat_bm25_by_ndcg": round(
            best["metrics"][f"ndcg@{args.metric_k}"] - bm25_metrics[f"ndcg@{args.metric_k}"],
            6,
        ),
        "notes": [
            "LSA dense backend is local TF-IDF + truncated SVD via SciPy; no model download and no server.",
            "Use this as a lightweight dense/hybrid hardening layer, not as a sentence-transformers result.",
        ],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
