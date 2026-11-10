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

## EU Central production canary

Commit `14b6e4c` retains request-local read reuse and vectorized PageRank.
It searches every query variant. The private canary used the live API image
with those two Python files replaced and shared its production datastore.
Only explicitly directed benchmark requests reached the canary. Shared chat
continued to use Quick.

The comparison alternated request order over 16 Balanced pairs. Production
ingestion was active during the window. One documentation pair had two
active worker jobs immediately before and after it. Most per-pair worker
checks timed out, so the full table describes the production test window.

| Brain | Pairs | Live API p50 | Canary p50 | Live API slowest | Canary slowest |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainAPI Developer Documentation | 10 | 25.231 s | 22.179 s | 31.690 s | 29.791 s |
| EP Signal Knowledge Base | 3 | 29.443 s | 19.903 s | 39.072 s | 26.593 s |
| BrainAPI Business Strategy | 3 | 8.887 s | 6.335 s | 9.844 s | 8.401 s |

The nearest-rank p95 equals the slowest response at these sample sizes.
All 32 requests succeeded, and all pairs returned identical ordered passages,
facts, and fact source IDs. Median documentation PageRank fell from 5.298 s
to 0.600 s. Searching all variants still took a median 20.730 s on the
canary, and six of ten documentation responses took at least 20 s.

Three canary process restarts reached HTTP readiness in 77.748 s, 72.759 s,
and 81.170 s. The first documentation request after each restart took
16.134 s, 12.418 s, and 15.936 s respectively. All three used ten query
variants and returned the same ordered passages as the initial canary run.
The tests used the existing database and caches throughout.

## Full-expansion fanout canary

Commit `1ae49eb` indexed graph edges by flow within each traversal and
ran passage searches in batches of up to three, preserving fusion order.
An opt-in EU Central canary used the production datastore. The paired
comparison used `14b6e4c` as the baseline in the same window.

| Brain | Baseline requests | Canary requests | Baseline p95 | Canary p95 |
| --- | ---: | ---: | ---: | ---: |
| BrainAPI Developer Documentation | 10 | 20 | 17.196 s | 16.873 s |
| EP Signal Knowledge Base | 3 | 3 | 15.489 s | 13.695 s |
| BrainAPI Business Strategy | 3 | 3 | 5.296 s | 5.145 s |

All 42 requests succeeded. All 16 pairs matched ordered passages, facts,
and fact source IDs. Every run searched the full set of query variants.
Worker checks observed ingestion in all 20 documentation sample windows.
The slowest documentation canary request took 19.593 s. The same-window
baseline also met the target. The graph neighbor stage fell from 3.036 s
to 2.533 s at the documentation median, while passage collection rose
from 1.345 s to 6.877 s. Total documentation fanout fell from 9.964 s to
9.581 s. Results are retrieval-only and represent one production load
window. Shared chat kept Quick. A separate generation smoke test returned
text from AWS Bedrock `us.anthropic.claude-haiku-5-5`.
