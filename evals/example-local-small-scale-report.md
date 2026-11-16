# Example local/small-scale report (not staging)

This file is a checked-in copy of a **local/small-scale** hybrid-only run against the in-repo fixtures (`--answerer extractive`, hashed n-gram dense, lexical reranker, no BrainAPI, no LLM keys). It is a retrieval ceiling on a tiny synthetic corpus. **Do not quote it externally as product quality.** Wilson intervals are wide. BrainAPI was not compared, so the head-to-head verdict is `incomplete`.

Regenerate with:

```bash
./evals.sh evaluate --use-case versioned_rules,policy_conflicts,expert_knowhow,decision_precedent \
  --split test --systems hybrid --include-optional --run local-test-hybrid --answerer extractive
```

---

# Use-case eval report `local-test-hybrid`

- Scale: **local/small-scale**
- Split: `test`
- Answerer: `extractive`
- Extractive scoring is a **retrieval ceiling** (gold facts in the packed window), not LLM answer quality. Use `--answerer llm` for the shared generative answerer.
- Prompt budget: `{"max_passages": 8, "max_chars": 6000, "max_answer_tokens": 256, "temperature": 0.0}`
- Answer prompt sha256: `095cb9e41f9f5b44e1b78db61ab11de9330114fa6785f9fdf732e59a37dd14e9`

No claim that BrainAPI beats hybrid search unless the numbers show it on the same scored items with non-overlapping Wilson intervals. Capability-pending cells are not losses.

Local/small-scale numbers are not a staging result and must not be quoted as product quality.

## Environment (redacted)

```json
{
  "brainapi_url": "http://localhost:8000",
  "brainpat_token": "",
  "answer_model": "deepseek-v4-flash",
  "judge_model": "deepseek-v4-flash",
  "embedding_provider": "hash",
  "embedding_model": "text-embedding-3-large",
  "reranker": "lexical",
  "scale": "local/small-scale",
  "as_of_field": "",
  "conflict_path": "",
  "llm_key_present": "no",
  "embedding_key_present": "no"
}
```

## versioned_rules

Documents: 13. Questions (test): 8.
Head-to-head: **incomplete** — Both systems must complete the same items before a win is claimed.

| system | accuracy | Wilson 95% | p50 e2e | p95 e2e | ingest | tokens | LLM calls | capability notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_baseline | 100.0% | 67.6%–100.0% | 0 ms | 1 ms | 9 ms | 0 | 0 | — |

Hybrid tuning (dev split only):

```json
{
  "best_config": {
    "chunk_size": 500,
    "chunk_overlap": 80,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fusion": "rrf",
    "rrf_k": 10,
    "dense_weight": 0.35,
    "retrieve_k": 20,
    "rerank_depth": 8,
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
    "conflict_threshold": 0.5
  },
  "best_dev_score": 1.0,
  "n_trials": 96
}
```

Top trials:

| fusion | chunk | k | rerank | dev score |
| --- | --- | --- | --- | --- |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| weighted | 500 | 20 | 16 | 1.000 |
| weighted | 500 | 20 | 8 | 1.000 |
| weighted | 500 | 40 | 16 | 1.000 |
| weighted | 500 | 40 | 8 | 1.000 |

## policy_conflicts

Documents: 14. Questions (test): 4.
Head-to-head: **incomplete** — Both systems must complete the same items before a win is claimed.

| system | accuracy | Wilson 95% | p50 e2e | p95 e2e | ingest | tokens | LLM calls | capability notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_baseline | 100.0% (pair F1 1.00) | 51.0%–100.0% | 11 ms | 11 ms | 5 ms | 0 | 0 | — |

Hybrid tuning (dev split only):

```json
{
  "best_config": {
    "chunk_size": 500,
    "chunk_overlap": 80,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fusion": "rrf",
    "rrf_k": 10,
    "dense_weight": 0.35,
    "retrieve_k": 20,
    "rerank_depth": 8,
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
    "conflict_threshold": 0.5
  },
  "best_dev_score": 0.09446164430684555,
  "n_trials": 96
}
```

Top trials:

| fusion | chunk | k | rerank | dev score |
| --- | --- | --- | --- | --- |
| rrf | 500 | 20 | 16 | 0.094 |
| rrf | 500 | 20 | 8 | 0.094 |
| rrf | 500 | 40 | 16 | 0.094 |
| rrf | 500 | 40 | 8 | 0.094 |
| rrf | 500 | 20 | 16 | 0.094 |
| rrf | 500 | 20 | 8 | 0.094 |
| rrf | 500 | 40 | 16 | 0.094 |
| rrf | 500 | 40 | 8 | 0.094 |
| weighted | 500 | 20 | 16 | 0.094 |
| weighted | 500 | 20 | 8 | 0.094 |
| weighted | 500 | 40 | 16 | 0.094 |
| weighted | 500 | 40 | 8 | 0.094 |

## expert_knowhow

Documents: 8. Questions (test): 6.
Head-to-head: **incomplete** — Both systems must complete the same items before a win is claimed.

| system | accuracy | Wilson 95% | p50 e2e | p95 e2e | ingest | tokens | LLM calls | capability notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_baseline | 100.0% | 61.0%–100.0% | 1 ms | 1 ms | 7 ms | 0 | 0 | — |

Hybrid tuning (dev split only):

```json
{
  "best_config": {
    "chunk_size": 500,
    "chunk_overlap": 80,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fusion": "rrf",
    "rrf_k": 10,
    "dense_weight": 0.35,
    "retrieve_k": 20,
    "rerank_depth": 8,
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
    "conflict_threshold": 0.5
  },
  "best_dev_score": 1.0,
  "n_trials": 96
}
```

Top trials:

| fusion | chunk | k | rerank | dev score |
| --- | --- | --- | --- | --- |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| weighted | 500 | 20 | 16 | 1.000 |
| weighted | 500 | 20 | 8 | 1.000 |
| weighted | 500 | 40 | 16 | 1.000 |
| weighted | 500 | 40 | 8 | 1.000 |

## decision_precedent

Documents: 5. Questions (test): 2.
Head-to-head: **incomplete** — Both systems must complete the same items before a win is claimed.

| system | accuracy | Wilson 95% | p50 e2e | p95 e2e | ingest | tokens | LLM calls | capability notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_baseline | 100.0% | 34.2%–100.0% | 0 ms | 0 ms | 2 ms | 0 | 0 | — |

Hybrid tuning (dev split only):

```json
{
  "best_config": {
    "chunk_size": 500,
    "chunk_overlap": 80,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fusion": "rrf",
    "rrf_k": 10,
    "dense_weight": 0.35,
    "retrieve_k": 20,
    "rerank_depth": 8,
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
    "conflict_threshold": 0.5
  },
  "best_dev_score": 1.0,
  "n_trials": 96
}
```

Top trials:

| fusion | chunk | k | rerank | dev score |
| --- | --- | --- | --- | --- |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| rrf | 500 | 20 | 16 | 1.000 |
| rrf | 500 | 20 | 8 | 1.000 |
| rrf | 500 | 40 | 16 | 1.000 |
| rrf | 500 | 40 | 8 | 1.000 |
| weighted | 500 | 20 | 16 | 1.000 |
| weighted | 500 | 20 | 8 | 1.000 |
| weighted | 500 | 40 | 16 | 1.000 |
| weighted | 500 | 40 | 8 | 1.000 |

## Judge agreement (hand-labeled sample)

- n = 8
- Cohen's kappa (exact-match vs human): 0.0
- Cohen's kappa (LLM judge vs human): not run

Hand-labeled sample in datasets/judge_agreement.jsonl. Kappa vs exact-match is reported even without an LLM judge. Pass --judge to also score the LLM judge on this sample.

## What this run is not

- Not a claim about staging or production quality unless `scale` says so and the BrainAPI target is the staging URL.
- Not a claim that BrainAPI beats hybrid search when the verdict is `undecided` or `incomplete`.
- Use cases `versioned_rules` and `policy_conflicts` need product as-of / conflict features before BrainAPI numbers are decisive.
