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

## Real brain staging results

On November 10, 2026, the isolated EU Central staging host ran the production
retrieval baseline (`152cf23`) and candidate (`5dded41`) on the same snapshots of
three real brains. Both builds used the production English spaCy model and
Vertex Gemini embeddings. For each question, 20 paired HTTP calls to
`/retrieve/context` alternated which build ran first. The Balanced preset
requested 40 facts, eight passages, and PageRank. The nearest-rank p95 includes
the first request on each build.

| Brain | Baseline p50 | Candidate p50 | Baseline p95 | Candidate p95 | Candidate max |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainAPI Developer Documentation | 11.454 s | 7.731 s | 12.961 s | 9.319 s | 27.503 s |
| EP Signal Knowledge Base | 10.719 s | 6.085 s | 12.688 s | 7.223 s | 7.817 s |
| BrainAPI Business Strategy | 4.811 s | 4.335 s | 5.960 s | 5.429 s | 5.801 s |

All 120 requests succeeded. All 60 pairs returned identical ordered passages,
facts, and fact source IDs. The documentation and EP Signal responses returned
eight passages and 40 facts each. The strategy response returned one passage
and 25 facts on both builds. Documentation PageRank fell from 4.097 s to
0.421 s at the median. All three questions still used full fact expansion, so
the real-data run did not exercise the early-stop branch.

The documentation run used the original question at the top of this report.
The other questions were “What is EP Signal and how does it help its users?”
and “How does BrainAPI support developers and teams building AI products?”
The EP Signal source passages did not contain the product name verbatim in
either build, so that pair measures evidence parity with the baseline.

## Production gate

All three staging p95s were under 20 s. The first documentation request still
exceeded 20 s on both builds (31.790 s baseline and 27.503 s candidate). The
benchmark measured retrieval over HTTP in isolation from live ingestion and
answer generation. Production shared chat remains on Quick.
