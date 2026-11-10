# Balanced context retrieval benchmark

The production documentation brain answered “How does BrainAPI store, organize,
and retrieve memories and contextual information?” with eight passages and 40
facts in **24.3 s** under Balanced. Quick took **20.1 s** in the same paired
profile. Both ran ten fact query variants. Balanced spent 19.2 s in retrieval
fan-out and 4.4 s in PageRank reranking. Profiled stages may overlap.

## Local before and after

Run from the repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/benchmark_balanced_retrieval.py --iterations 15
```

The replay passes the exact question through the context controller using a
fixed documentation fixture and simulated storage contention. It does not
access the production brain, run answer generation, or measure production
latency. The before run used the clean upstream base for this change.

| Measure | Before | After |
| --- | ---: | ---: |
| Controller p50 | 321 ms | 75 ms |
| Controller p95 | 420 ms | 82 ms |
| Fact embedding / entity vector / relationship vector calls per turn | 10 / 10 / 10 | 1 / 1 / 1 |
| Expected fixture passages returned, in the same order | 8 / 8 | 8 / 8 |
| Facts returned | 40 | 40 |

The fixture returned the same ordered facts and passages in both builds.
PageRank keeps the full adjacency and runs in a worker thread. The tests check
that weak first-pass evidence still triggers expansion, repeated graph reads are
shared only within a request, and ranking runs outside the event loop.

## Production gate

Keep Quick and the 45-second shared-chat context deadline until the candidate
can be profiled against the same documentation data in a production-equivalent
environment. Compare repeated Balanced runs on the exact question and on
representative brains and questions, including ones that need expansion or
reranking. Check p95 against 20 seconds and review the actual sources, citations,
and no-answer rates. Verify the public chat with the candidate build before
restoring Balanced as the default. The local fixture does not satisfy this gate.
