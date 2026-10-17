# Production validation artifact contract

Run `scripts/check_release_readiness.py DIR` before creating an RC or stable
tag. `DIR` must contain the following files produced against the exact
candidate image:

- `smoke.json`: `light` and `heavy`, each with `unexpected_5xx: 0` and true
  flow flags for authentication, brain creation, plain/structured ingestion,
  task completion, context, search, Console, and MCP.
- `latency.json`: context/search p50, p95, and p99; context records zero online
  LLM retrieval loops; search records `excludes_embed_query: true`.
- `restore-light.json` and `restore-heavy.json`: matching `before`/`after`
  counts for brains, nodes, edges, chunks, vectors, and observations plus ten
  deterministic retrieval checks with `match: true`.
- `security.json`: `high: 0`, `critical: 0`, the `sha256:` image digest, and
  the relative path of its SBOM.

The public ledger must also contain a representative LongMemEval result and a
WANDS representative Search row. LoCoMo and BEAM rows retain their explicit
sample scopes and are not promoted into broader claims by this gate.

Dev tags (`vX.Y.Z-dev` / `vX.Y.Z-dev.N`) still require smoke, restore, and
security on the exact image. They skip the public-ledger and latency SLO
claims so a working image can publish before those product scores exist.
Pass `--tag` so the publish workflow can apply that distinction.

`scripts/production_smoke.py exercise` produces profile-scoped smoke, latency,
and pre-restore state evidence; `verify-restore` produces the matching restore
report. `scripts/assemble_release_artifacts.py` combines successful light and
heavy workflow artifacts. The tag-only publish workflow performs this assembly
and runs the gate itself, in addition to requiring all quality and dedicated
heavy-runner checks on the tagged commit.
