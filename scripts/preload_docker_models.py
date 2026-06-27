from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SENTENCE_TRANSFORMER_MODELS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/e5-small",
)

MAX_ATTEMPTS = 8
BASE_DELAY_SECONDS = 5


def _is_rate_limited(exc: BaseException) -> bool:
    if "429" in str(exc):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _retry(action, label: str) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            action()
            return
        except Exception as exc:
            if attempt == MAX_ATTEMPTS or not _is_rate_limited(exc):
                raise
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"[preload] {label}: rate limited, "
                f"retry {attempt}/{MAX_ATTEMPTS} in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def _configure_hf_token() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _preload_sentence_transformers() -> None:
    from sentence_transformers import SentenceTransformer

    for model in SENTENCE_TRANSFORMER_MODELS:
        _retry(lambda model=model: SentenceTransformer(model), model)


def _preload_spacy() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.constants.spacy_models import SPACY_MODEL_NAMES

    for model in sorted(set(SPACY_MODEL_NAMES.values())):

        def download(model_name: str = model) -> None:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"spacy download {model_name} failed with exit code {result.returncode}"
                )

        _retry(download, model)


def main() -> None:
    _configure_hf_token()
    _preload_sentence_transformers()
    _preload_spacy()


if __name__ == "__main__":
    main()
