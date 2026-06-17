# CONTEXT - sport-health-search-foundry

## Goals

- Build a local-first sports/health retrieval prototype on BEIR NFCorpus with deterministic indexing, search, and baseline evaluation.
- Keep Phase 1 CI serverless: no OpenSearch, no Docker, no external service dependency.
- Expose a stable `search(query) -> ranked doc_ids` interface for later OpenSearch, dense, or reranker backends.
- Use official NFCorpus qrels for objective retrieval metrics instead of hand-labeled Phase 1 qrels.

## Non-Goals

- No raw NFCorpus data, generated indexes, or internal tooling in the public tree.
- No private user research data.
- No OpenSearch/Qdrant service in Phase 1; those are later backends after the golden eval contract is stable.
- No claim of production search-platform coverage until official backend parity is tested.

## Data Assumptions

- NFCorpus is fetched from the public BEIR dataset distribution and pinned by URL, `as_of`, and hash in a local manifest.
- CI uses synthetic NFCorpus-style golden fixtures, not network-fetched benchmark data.
- Baseline reproduction depends on analyzer/tokenization; Phase 1 local BM25 is a transparent baseline, not a claim of exact Anserini/OpenSearch parity unless the score matches official baseline range.

## Architecture

- `scripts/fetch_nfcorpus.py` downloads and extracts BEIR NFCorpus into `data/raw/nfcorpus/`.
- `src/shsf/bm25.py` implements a dependency-free BM25 index.
- `scripts/build_index.py` builds a deterministic JSON index from JSONL documents.
- `scripts/search.py` runs command-line search over an index.
- `scripts/evaluate_bm25.py` runs NDCG@10 / Recall@10 / MRR@10 over official qrels.
- `scripts/smoke.py` builds a tiny golden index and asserts deterministic top-ranked output.

## Known Limitations

- BM25 only; no dense retrieval, reranker, OpenSearch, or hybrid scoring service yet.
- Golden fixture smoke proves deterministic retrieval plumbing, not domain relevance quality.
- Matching BEIR official BM25 exactly may require Pyserini/Anserini or OpenSearch analyzer parity in a later phase.
