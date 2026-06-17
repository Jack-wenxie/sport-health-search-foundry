# Dataset Card: BEIR NFCorpus phase1

## Overview
- Source: BEIR public benchmark dataset distribution.
- Dataset: NFCorpus.
- Collection script: `scripts/fetch_nfcorpus.py`.
- Local root: `data/raw/nfcorpus/`.
- Manifest: `data/raw/nfcorpus.manifest.json`.
- Downloaded as_of: 2026-06-14T12:23:14+00:00.
- ZIP SHA-256: `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`.
- Status: private-local benchmark working copy; safe for local evaluation, public release still requires pre-publish review.

## Source / License / Access
- Download URL is pinned in `scripts/fetch_nfcorpus.py`.
- NFCorpus is a public benchmark used by BEIR.
- License terms should be checked from the downloaded dataset metadata / upstream before public redistribution.
- This repo remains private-first; no GitHub push before user approval and privacy/license review.

## Fields
- `corpus.jsonl`: BEIR corpus records with document id, title, and text.
- `queries.jsonl`: BEIR query records.
- `qrels/*.tsv`: official relevance judgments by split.

## Collection And Cleaning
- The fetch script downloads the official ZIP, extracts it, and records SHA-256 hash, as_of timestamp, file paths, and detected qrels splits.
- No manual qrels are created in Phase 1.
- BM25 indexing reads the official corpus JSONL directly.

## Bias / Privacy / Leakage
- No private user data.
- Evaluation is limited to NFCorpus task definitions and qrels.
- Do not mix qualitative sport-science showcase queries into objective NFCorpus metrics.

## Intended Use
- Reproducible biomedical retrieval baseline.
- BM25 baseline for NDCG@10 / Recall@10 / MRR@10.
- Future comparison with OpenSearch, dense retrieval, reranking, and domain-specific sport-health queries.

## Non-Use
- Not a clinical evidence synthesis.
- Not a claim of large-scale production search.
- Not an objective sport-science benchmark beyond NFCorpus unless a separate qrels protocol is added.
