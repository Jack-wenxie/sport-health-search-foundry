from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shsf.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from shsf.lsa import LsaIndex, weighted_fusion

DOCS = ROOT / "tests" / "golden" / "golden_docs.jsonl"
INDEX = ROOT / "tests" / "golden" / "smoke_index.json"
VERIFY_RECORD = ROOT / "benchmarks" / "raw" / "verification-smoke-latest.json"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)


def main() -> int:
    run([sys.executable, "scripts/build_index.py", "--input", str(DOCS), "--output", str(INDEX)])
    out = run(
        [
            sys.executable,
            "scripts/search.py",
            "--index",
            str(INDEX),
            "--query",
            "knee osteoarthritis gait biomechanics",
            "--top-k",
            "3",
        ]
    )
    rows = json.loads(out.stdout)
    if not rows:
        raise SystemExit("smoke failed: no results")
    if rows[0]["doc_id"] != "G001":
        raise SystemExit(f"smoke failed: expected G001 top-1, got {rows[0]['doc_id']}")
    ranked = [row["doc_id"] for row in rows]
    rels = {"G001": 2, "G003": 1}
    ndcg10 = ndcg_at_k(ranked, rels, 10)
    recall10 = recall_at_k(ranked, rels, 10)
    mrr10 = mrr_at_k(ranked, rels, 10)
    if ndcg10 <= 0 or recall10 <= 0 or mrr10 <= 0:
        raise SystemExit("smoke failed: mini metric calculation returned zero")
    from shsf.corpus import read_jsonl
    from shsf.bm25 import BM25Index

    docs = read_jsonl(DOCS)
    bm25 = BM25Index.build(docs)
    lsa = LsaIndex.build(docs, dims=2, max_features=100, min_df=1)
    fused = weighted_fusion(
        bm25.search("wearable gait parkinson sensors", top_k=3),
        lsa.search("wearable gait parkinson sensors", top_k=3),
        dense_weight=0.2,
        top_k=3,
    )
    if not fused:
        raise SystemExit("smoke failed: hybrid fusion returned no results")
    VERIFY_RECORD.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_RECORD.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "python": "python",
                "smoke": "scripts/smoke.py",
                "golden_fixture": str(DOCS.relative_to(ROOT)),
                "index_path": str(INDEX.relative_to(ROOT)),
                "query": "knee osteoarthritis gait biomechanics",
                "expected_top1": "G001",
                "actual_top1": rows[0]["doc_id"],
                "mini_metrics": {
                    "ndcg@10": round(ndcg10, 6),
                    "recall@10": round(recall10, 6),
                    "mrr@10": round(mrr10, 6),
                },
                "top_k": rows,
                "hybrid_smoke_top1": fused[0].doc_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("smoke ok: deterministic BM25, mini metrics, and hybrid fusion passed")
    print(f"verification_record={VERIFY_RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
