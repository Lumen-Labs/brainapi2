# Expert know-how capture

Task: answer procedural, multi-hop questions that require combining expert-written procedures and notes (runbooks, rotas, topology, freeze calendar, lab chain-of-custody).

This is the use case that is **runnable on current main**: ingest text, `POST /retrieve/context`, shared LLM/extractive answerer. No as-of API or conflict endpoint is required.

## Provenance

Original **Northwind SRE / lab** procedures written for this harness. They are realistic in tone (pager rota, shard map, Sev-2 steps, change freeze, sample custody) but are not copied from a vendor's private runbooks.

## Licensing

CC0 1.0 (original work in this repository).

## Split

`smoke` is two multi-hop items for CI. Tune retrieval hyperparameters on `dev`. Report `test`. Gold uses short answers plus `must_include` so scoring is not prompt-fitted to a single gold string.

## Scoring

`must_include` of the key entities/steps. Optional LLM judge with the fixed rubric when `--answerer llm --judge` is set. A hand-labeled sample lives in `datasets/judge_agreement.jsonl`.
