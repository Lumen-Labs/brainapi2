# Expert know-how capture

Task: answer procedural, multi-hop questions that require combining expert-written procedures (public runbooks/SOPs).

This is the use case that is **runnable on current main**: ingest text, `POST /retrieve/context`, shared LLM/extractive answerer. No as-of API or conflict endpoint is required.

## Provenance

**Scored split:** [prometheus-operator/runbooks](https://github.com/prometheus-operator/runbooks) (Apache-2.0). Each item pairs two runbooks in the same component (or a runbook and a markdown-linked peer). Gold `must_include` spans are the first mitigation sentence of one runbook and the first diagnosis/meaning sentence of the other. See [`GOLD.md`](GOLD.md).

**Smoke split:** original Northwind SRE / lab procedures (`ci_smoke: true`).

## Licensing

- Scored runbooks: Apache-2.0 (prometheus-operator/runbooks).
- Smoke fixtures: CC0 1.0 (original work in this repository).

## Split

`smoke` is two multi-hop items for CI. Tune retrieval hyperparameters on `dev`. Report `test` (≥150 scored items). Gold uses short `must_include` spans so scoring is not prompt-fitted to a single gold string.

## Scoring

`must_include` of the key entities/steps. Optional LLM judge with the fixed rubric when `--answerer llm --judge` is set. A hand-labeled sample lives in `datasets/judge_agreement.jsonl`.
