# Evidence Matrix

| Evidence ID | Claim | Artifact path | Capability | Evidence tier | Verification command | Raw log / result path | Reproducibility status | Counterexample / failure test | External-validation target | Last verified | owner | Upgrade blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Local BM25 retrieval is evaluated on NFCorpus official qrels. | `benchmarks/raw/nfcorpus-bm25-baseline.json` | Information retrieval evaluation | P1-adjacent | `scripts/verify.ps1` + `scripts/evaluate_bm25.py` | `benchmarks/raw/nfcorpus-bm25-baseline.json` | Reproducible with CI golden fixture | Smoke test asserts deterministic top-k output | Compare against a public Pyserini baseline | 2026-06-14 | local implementation + peer review | Exact analyzer parity with official backends |
| E2 | Weighted hybrid retrieval (BM25 + local LSA via TF-IDF + truncated SVD) improves over the BM25 baseline on NFCorpus: NDCG@10 0.311 to 0.329, Recall@10 0.152 to 0.164, and MRR 0.519 to 0.529. LSA is a lightweight dense signal, not a sentence-transformers model. | `benchmarks/raw/nfcorpus-hybrid-lsa-sweep.json` + `src/shsf/lsa.py` | Hybrid retrieval evaluation | P1-adjacent | `scripts/evaluate_hybrid_lsa.py` deterministic sweep | `benchmarks/raw/nfcorpus-hybrid-lsa-sweep.json`, best=`weighted-bm25-lsa-d256-w0.3` | Reproducible deterministic sweep | Dense-only LSA underperforms BM25; `dense_weight=0.5` drops performance; 24-config sweep is recorded | Add CI golden coverage and compare against an official backend | 2026-06-15 | local implementation + peer review | CI golden coverage plus Pyserini/OpenSearch analyzer comparison |

## Evidence Tier Rules

- Do not mark a claim as strong reproducible evidence unless it has a runnable verification command and a raw result or log path.
- Do not mark a claim as externally validated unless it has an external URL, merged PR, leaderboard result, citation, or third-party reproduction evidence.
- If verification is manual-only, mark it as review-required rather than reproducible.
