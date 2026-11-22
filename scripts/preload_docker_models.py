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

MAX_ATTEMPTS = 10
BASE_DELAY_SECONDS = 10
MAX_DELAY_SECONDS = 120


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_rate_limited(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        text = str(item).lower()
        if (
            "429" in text
            or "too many requests" in text
            or "rate limit" in text
            or "ratelimit" in text
        ):
            return True
    return False


def _is_transient_hub_error(exc: BaseException) -> bool:
    if _is_rate_limited(exc):
        return True
    for item in _exception_chain(exc):
        text = str(item).lower()
        if (
            "couldn't connect to 'https://huggingface.co'" in text
            or "localentrynotfounderror" in type(item).__name__.lower()
            or "temporarily unavailable" in text
            or "service unavailable" in text
            or "gateway timeout" in text
        ):
            return True
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _retry(action, label: str) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            action()
            return
        except Exception as exc:
            if attempt == MAX_ATTEMPTS or not _is_transient_hub_error(exc):
                raise
            retry_after = _retry_after_seconds(exc)
            exponential = min(
                BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                MAX_DELAY_SECONDS,
            )
            delay = max(retry_after or 0.0, float(exponential))
            kind = "rate limited" if _is_rate_limited(exc) else "hub error"
            print(
                f"[preload] {label}: {kind}, "
                f"retry {attempt}/{MAX_ATTEMPTS} in {delay:.0f}s "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            time.sleep(delay)


def _configure_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if not token:
        print(
            "[preload] warning: no HF_TOKEN/HUGGING_FACE_HUB_TOKEN set; "
            "anonymous Hugging Face downloads are frequently rate-limited in CI",
            flush=True,
        )
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    print("[preload] Hugging Face token configured", flush=True)
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"[preload] huggingface_hub.login skipped: {exc}", flush=True)


def _preload_sentence_transformers() -> None:
    from sentence_transformers import SentenceTransformer

    for model in SENTENCE_TRANSFORMER_MODELS:
        print(f"[preload] loading sentence-transformers model: {model}", flush=True)
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

        print(f"[preload] downloading spaCy model: {model}", flush=True)
        _retry(download, model)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    spacy_only = "--spacy-only" in args
    local_ml_only = "--local-ml-only" in args
    if spacy_only and local_ml_only:
        raise SystemExit("choose one of --spacy-only or --local-ml-only")
    if spacy_only:
        _preload_spacy()
        return
    if local_ml_only:
        _configure_hf_token()
        _preload_sentence_transformers()
        return
    _configure_hf_token()
    _preload_sentence_transformers()
    _preload_spacy()


if __name__ == "__main__":
    main()
