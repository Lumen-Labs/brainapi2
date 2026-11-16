# Policy contradiction audit

Task: from a help-center-style corpus, surface pairs of statements that a reader cannot follow at once for the same audience and time. Score precision/recall of unordered statement-id pairs.

## Provenance

Original **Northwind Cloud** help-center articles written for this harness. Four contradictions are **seeded** (refund window, retention, password length, 2FA). Two near-misses are **not** contradictions because they are scoped to different plans (standard vs enterprise support SLA) or to a clearly dated sunset. Remaining articles are internally consistent decoys.

This is not scraped from a live vendor help center.

## Licensing

CC0 1.0 (original work in this repository).

## Split

`gold_pairs.jsonl` is the full gold set. Questions in `questions.jsonl` ask the system to list conflicts (used for the QA view). Pair-level P/R is the headline metric and uses the gold file, not the free-text answer.

`smoke` / `dev` / `test` questions exist so the same runner works for QA and for pair detection. Tuning of the contradiction threshold uses `dev` questions plus the gold pairs listed on those rows.

## BrainAPI note

Current main may return `temporal_conflicts` on `/retrieve/context`, which is **not** statement-pair conflict detection. The adapter treats a future `EVAL_CONFLICT_PATH` as the real plugin. Until that lands, BrainAPI pair P/R is reported as capability-pending if no pairs are produced.
