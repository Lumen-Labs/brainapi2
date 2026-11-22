"""
File: /spacy.py
Project: nlp
Created Date: Sunday February 8th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 8th 2026 11:22:44 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import logging
import threading
import time

import spacy
from src.config import config
from src.constants.spacy_models import SPACY_MODEL_NAMES

MODEL_NAMES = SPACY_MODEL_NAMES
logger = logging.getLogger(__name__)

_UNAVAILABLE_LOG_INTERVAL_SECONDS = 60.0
_FAILED = object()


class SpacyManager:
    def __init__(self):
        self.keep_models_in_memory = config.spacy.keep_models_in_memory
        self._cache = {}
        self._errors = {}
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._unavailable_log_at = {}

    def _lock_for(self, lang: str):
        with self._locks_guard:
            lock = self._locks.get(lang)
            if lock is None:
                lock = threading.Lock()
                self._locks[lang] = lock
            return lock

    def _failed_error(self, lang: str) -> OSError:
        return OSError(self._errors[lang])

    def _raise_if_failed(self, lang: str, cached):
        if cached is _FAILED:
            raise self._failed_error(lang)

    def _log_unavailable(self, lang: str, model_name: str, exc: BaseException) -> None:
        now = time.monotonic()
        last = self._unavailable_log_at.get(lang)
        if last is not None and (now - last) < _UNAVAILABLE_LOG_INTERVAL_SECONDS:
            return
        self._unavailable_log_at[lang] = now
        logger.error(
            "SpaCy model %r failed to load for lang=%s: %s. "
            "Not downloading on the request path; NER returns empty for this language.",
            model_name,
            lang,
            exc,
        )

    def get_model(self, lang: str):
        if lang not in MODEL_NAMES:
            raise ValueError(f"Unsupported language: {lang}")
        cached = self._cache.get(lang)
        self._raise_if_failed(lang, cached)
        if self.keep_models_in_memory and cached is not None:
            return cached
        model_name = MODEL_NAMES[lang]
        with self._lock_for(lang):
            cached = self._cache.get(lang)
            self._raise_if_failed(lang, cached)
            if self.keep_models_in_memory and cached is not None:
                return cached
            return self._load_model(lang, model_name)

    def _load_model(self, lang: str, model_name: str):
        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            message = (
                f"SpaCy model {model_name!r} is not installed. "
                f"NER is disabled for lang={lang}."
            )
            self._errors[lang] = message
            self._cache[lang] = _FAILED
            self._log_unavailable(lang, model_name, exc)
            raise OSError(message) from None
        if self.keep_models_in_memory:
            self._cache[lang] = nlp
        return nlp

    def warmup(self, lang: str = "en") -> None:
        def _run() -> None:
            try:
                self.get_model(lang)
            except OSError:
                pass

        threading.Thread(
            target=_run,
            name=f"spacy-warmup-{lang}",
            daemon=True,
        ).start()

    def unload(self, lang: str):
        self._cache.pop(lang, None)
        self._errors.pop(lang, None)

    def unload_all(self):
        self._cache.clear()
        self._errors.clear()


_spacy_manager = SpacyManager()


def warmup_default_model(lang: str = "en") -> None:
    """Load the default NER pipeline in a background thread at process start."""
    _spacy_manager.warmup(lang)
