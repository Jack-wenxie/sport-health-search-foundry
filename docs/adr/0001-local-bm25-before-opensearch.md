# ADR-0001: Use BEIR NFCorpus + Local BM25 Before OpenSearch

- Status: Accepted
- Date: 2026-06-14

## Context

Phase 1 needs a reproducible sports and health retrieval baseline that can be compared against a public benchmark. A small hand-built PubMed corpus would make the first milestone depend on custom labeling, while BEIR NFCorpus already provides a public corpus, queries, and qrels.

OpenSearch is closer to a production search platform, but starting with a service stack would make the first milestone depend on service startup, index lifecycle management, and CI infrastructure. The first milestone should instead prove that the retrieval and evaluation contract is reproducible.

## Decision

Phase 1 uses BEIR NFCorpus plus a dependency-light local BM25 implementation:

- `scripts/fetch_nfcorpus.py` downloads and hashes NFCorpus.
- `src/shsf/bm25.py` provides deterministic tokenization, index building, and search.
- `scripts/evaluate_bm25.py` uses official qrels to emit NDCG@10 / Recall@10 / MRR@10 baseline JSON.
- `scripts/smoke.py` uses a synthetic golden fixture to assert the top-1 document for a fixed query.
- PubMed scraping is not used as the primary Phase 1 evidence path.
- OpenSearch, Qdrant, and hybrid retrieval are deferred until the qrels-backed evaluation protocol is stable.

## Consequences

- CI can validate the core interface without a server or network dependency.
- NFCorpus provides public qrels and historical baselines, so Phase 1 does not depend on custom relevance labels.
- The retrieval interface is stable enough for later OpenSearch, dense retrieval, and reranking backends.
- Phase 1 should not claim OpenSearch, Kubernetes, or distributed search coverage.
- The local BM25 analyzer may not match official BEIR Anserini/Pyserini/OpenSearch baselines; any score gap must be reported as an analyzer limitation.
