# Versioned rules / as-of

Task: answer questions of the form "what did clause X say as of date D" and "what changed between D1 and D2" over a small regulation corpus with dated versions.

## Provenance

The in-repo fixture is **original evaluation text** written for this harness. It follows the *shape* of the U.S. Federal Acquisition Regulation (clause ids, dated revisions, inspection/payment/option language) but is **not** an excerpt of FAR or eCFR. Do not cite it as current law.

A download helper (`download_far.py`) can fetch **public-domain** U.S. government FAR HTML from [acquisition.gov](https://www.acquisition.gov/) for a larger run. Those files are U.S. government works and are not copyrighted in the United States (see 17 U.S.C. § 105). The helper stores fetch timestamps and source URLs; it does **not** auto-label gold as-of answers. Historical versions still need a human or eCFR-history pass before they are scored.

## Licensing

- Fixture (`corpus.jsonl`, `questions.jsonl`): CC0 1.0 (original work in this repository).
- Downloaded FAR HTML: U.S. government public domain. Keep source URLs in the sidecar manifest.

## Split

Questions are tagged `smoke` / `dev` / `test`. Tune fusion and chunking on `dev` only. Report `test`. `smoke` is a disjoint wiring set for CI.

## Scoring

Short answers use exact match (normalized) plus `must_include` phrases. Changelog items require every `must_include` phrase.

## BrainAPI note

Current main has no `as_of` query parameter. The BrainAPI adapter still runs these questions as written (the date is in the question text) and marks `as_of` capability pending unless OpenAPI/env exposes a field.
