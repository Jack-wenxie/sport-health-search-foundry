# sport-health-search-foundry

> Phase 1: local, deterministic BM25 retrieval and evaluation on BEIR NFCorpus.

## What This Project Does

This repository builds a local sports and health retrieval baseline on the BEIR NFCorpus benchmark. It creates a deterministic BM25 index, evaluates it with the official qrels, reports NDCG@10 / Recall@10 / MRR@10, and exposes a `search(query) -> ranked doc_ids` interface.

Phase 1 keeps the stack local and CI-friendly before adding service-backed retrieval such as OpenSearch or dense reranking.

## Evidence Status

Phase 1 provides a reproducible benchmark manifest, BM25 baseline JSON, golden smoke fixture, and verification record. The current implementation is a transparent local baseline; exact parity with Pyserini/Anserini or OpenSearch analyzers is a later validation step.

## How To Run

```powershell
./scripts/verify.ps1
```

Fetch NFCorpus and run the local BM25 baseline. This path uses the network and is intentionally separate from the CI smoke test:

```powershell
python scripts/fetch_nfcorpus.py
python scripts/evaluate_bm25.py --split test
```

Current local BM25 baseline on NFCorpus test split (regex analyzer):

- documents: 3633
- queries evaluated: 323
- NDCG@10: 0.310981
- Recall@10: 0.152029
- MRR@10: 0.519211
- output: `benchmarks/raw/nfcorpus-bm25-baseline.json`
