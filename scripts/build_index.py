from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shsf.bm25 import BM25Index
from shsf.corpus import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    docs = read_jsonl(args.input)
    if not docs:
        raise SystemExit(f"No documents found: {args.input}")
    index = BM25Index.build(docs)
    index.save(args.output)
    print(f"indexed_docs={len(index.doc_ids)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
