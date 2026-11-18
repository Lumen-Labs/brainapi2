# Decision precedent (optional)

Task: given a small log of past incidents and the decisions taken, answer "what did we decide last time a similar case happened?"

Included because the corpus is tiny and the scorer is the same as expert know-how. Skip with `--use-case` lists that omit `decision_precedent`.

## Provenance

Original Northwind incident-decision notes written for this harness.

## Licensing

CC0 1.0.

## Scoring

`must_include` on the prior decision and ticket id.
