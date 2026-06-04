# brainapi

Interactive installer and runtime CLI for [BrainAPI](https://github.com/Lumen-Labs/brainapi2).

## Install

```bash
npm install -g brainapi-tui
```

This puts a `brainapi` binary on your `$PATH`.

## Quick start

```bash
brainapi init    # clone the project, set up a Python venv, configure services
brainapi start   # start docker compose services + run the API
brainapi doctor  # check Python, Docker, Ollama, and configured services
```

## Commands

| Command           | Description                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `brainapi init`   | Full bootstrap: clone the repo, create a Python venv, install deps, and run the interactive setup. |
| `brainapi start`  | Start docker compose containers for the chosen services and launch the API.                        |
| `brainapi config` | Re-run the interactive flow and rewrite `.env`.                                                    |
| `brainapi doctor` | Check that Python, Docker, Ollama, GCP credentials, and configured services are reachable.         |
| `brainapi update` | `git pull` the project and `pip install -e .` again.                                               |

### `brainapi start` options

| Option                             | Description                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--pipeline accurate\|lightweight` | Set `PIPELINE_MODE` in `~/.brainapi/source/.env` before start (accurate = full pipeline; lightweight = faster ingestion). |

## Where things live

- Source: `~/.brainapi/source/`
- Python venv: `~/.brainapi/source/.venv/`
- Env file: `~/.brainapi/source/.env`
- Install state: `~/.brainapi/state.json`

## Configuration

The interactive flow asks:

1. Use default settings? (NetworkX + Postgres + pgvector + remote GCP Vertex)
2. Otherwise: Vector DB → Data DB → Graph DB → Models mode
3. For `remote` models mode: pick provider, then provide GCP credentials, project, and model names
4. For `local` models mode: probe Ollama, wait for it to start, then verify pulled models
5. Connection details for only the services you actually selected
6. `BRAINPAT_TOKEN` (generate or paste)
7. Optionally start the docker compose containers now

If Python (>=3.11) or Docker is missing, the TUI walks you through installing them with platform-aware commands and retries detection automatically.

## Development

```bash
cd tui
npm install
npm run dev    # builds in watch mode
node dist/index.js init   # try it
```

### Publishing

```bash
npm version <patch|minor|major>
npm publish
```

`prepublishOnly` runs the build automatically so the published tarball always contains `dist/`. Only the `dist/`, `README.md`, and `LICENSE` files are included (see the `files` field in `package.json`).

If someone runs any command before `brainapi init` (e.g. `brainapi doctor` or `brainapi start`), the CLI prints a notice and runs `init` first.

## Environment overrides

| Variable            | Default                                       | Description                       |
| ------------------- | --------------------------------------------- | --------------------------------- |
| `BRAINAPI_REPO_URL` | `https://github.com/Lumen-Labs/brainapi2.git` | The git repo cloned on `init`.    |
| `BRAINAPI_HOME`     | `$HOME/.brainapi`                             | Where the source and state live.  |
| `BRAINAPI_BRANCH`   | `main`                                        | Branch to checkout after cloning. |
