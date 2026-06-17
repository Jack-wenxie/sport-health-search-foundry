from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shsf.bm25 import BM25Index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index = BM25Index.load(args.index)
    rows = [r.__dict__ for r in index.search(args.query, top_k=args.top_k)]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
