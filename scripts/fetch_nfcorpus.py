from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"


def https_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "sport-health-search-foundry/phase1"})
    with urllib.request.urlopen(req, timeout=180, context=https_context()) as resp:
        with output.open("wb") as f:
            shutil.copyfileobj(resp, f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=NFCORPUS_URL)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "nfcorpus.zip"
    dataset_dir = args.out_dir / "nfcorpus"
    if args.force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if args.force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        download(args.url, zip_path)

    if not dataset_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(args.out_dir)

    root = dataset_dir
    if not root.exists():
        # Some ZIP tools can create a nested directory with a different case.
        candidates = [p for p in args.out_dir.iterdir() if p.is_dir() and p.name.lower() == "nfcorpus"]
        if not candidates:
            raise SystemExit("NFCorpus extraction did not create data/raw/nfcorpus")
        root = candidates[0]

    qrels_dir = root / "qrels"
    qrels = sorted(p.name for p in qrels_dir.glob("*.tsv")) if qrels_dir.exists() else []
    manifest = {
        "name": "beir_nfcorpus",
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_url": args.url,
        "zip_path": str(zip_path.relative_to(ROOT)),
        "dataset_dir": str(root.relative_to(ROOT)),
        "zip_sha256": sha256_file(zip_path),
        "files": {
            "corpus": str((root / "corpus.jsonl").relative_to(ROOT)),
            "queries": str((root / "queries.jsonl").relative_to(ROOT)),
            "qrels": qrels,
        },
        "license_note": "NFCorpus is a public BEIR benchmark. Check upstream dataset metadata/license before public redistribution.",
    }
    manifest_path = args.out_dir / "nfcorpus.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
