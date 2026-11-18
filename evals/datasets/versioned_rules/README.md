# Versioned rules / as-of

Task: answer "what did clause X say as of date D" and "what changed between D1 and D2" over historical FAR clause text.

## Provenance

**Scored split (dev/test):** U.S. Federal Acquisition Regulation as published in eCFR Title 48 Chapter 1. Snapshots are fetched from the eCFR Versioner API (`/api/versioner/v1/full/{date}/title-48.xml?chapter=1&part=…`). Gold answers are derived **reproducibly from section-level diffs** (see [`GOLD.md`](GOLD.md) and `far_fetch.json`). FAMR fixtures are **not** in the scored split.

**Smoke split:** original FAMR evaluation text (`ci_smoke: true`). Shape-compatible, not an excerpt of FAR. Used only by CI.

Rebuild:

```bash
PYTHONPATH=. python3 -m evals.datasets.build_public_corpora
```

## Licensing

- Scored FAR/eCFR excerpts: U.S. government work, 17 U.S.C. § 105 (public domain in the U.S.). Source URLs are stored on each document.
- Smoke FAMR fixtures: CC0 1.0 (original work in this repository).

## Split

`smoke` is a tiny disjoint wiring set. Tune on `dev` only. Report `test` (≥150 scored items). Spot-check sample: `spotcheck.jsonl`.

## Scoring

Short answers use exact match plus `must_include` phrases. Changelog items require every `must_include` phrase.

## BrainAPI note

Current main has no `as_of` query parameter. The BrainAPI adapter feature-detects an as-of field when OpenAPI/`EVAL_AS_OF_FIELD` exposes one. Until that lands it sends `include_history=true` for as-of and changelog questions and records `as_of_mode` on each result.
