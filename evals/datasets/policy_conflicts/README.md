# Policy contradiction audit

Task: from a help-center / policy corpus, surface pairs of statements that a reader cannot follow at once for the same audience and time. Score precision/recall of unordered statement-id pairs.

## Provenance

**Scored split:** real public help-center and policy text.

- GitHub Docs (`github/docs`, CC-BY-4.0) — authentication, billing, org security, REST intro.
- GSA-TTS handbook (`GSA-TTS/handbook`, U.S. government work / CC0) — travel, leave, tech policies.

Gold pairs are mostly **seeded** (one documented numeric or required/optional mutation of a real paragraph, stored as a "stale copy") plus a smaller set of **naturally occurring** numeric clashes between unmodified paragraphs that share title topic tokens. Plan-scoped Standard vs Enterprise near-misses are `non_pairs`, not gold. See [`GOLD.md`](GOLD.md).

**Smoke split:** original Northwind Cloud articles (`ci_smoke: true`).

## Licensing

- GitHub Docs excerpts: CC-BY-4.0; attribution: [github/docs](https://github.com/github/docs).
- GSA-TTS handbook excerpts: U.S. government work / CC0.
- Seeded mutations and Northwind smoke fixtures: original annotations in this repository (CC0).

## Split

`gold_pairs.jsonl` is the full gold set (≥150 pairs). Pair-level P/R is the headline metric. `smoke` / `dev` / `test` questions exist so the same runner works for QA and for pair detection. Tune on `dev` only.

## BrainAPI note

Current main may return `temporal_conflicts` on `/retrieve/context` as graph hub/entity UUIDs, which are not statement-pair gold. The adapter maps those ids back to ingested `statement_id` / document ids (headers at ingest, `meta_keys`, retrieve metadata) before scoring. If mapping is impossible, pair F1 is marked **unsupported** rather than 0. A future `EVAL_CONFLICT_PATH` is still used when present.
