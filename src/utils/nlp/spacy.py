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
from src.constants.spacy_models import SPACY_MODEL_NAMES, resolve_spacy_lang

MODEL_NAMES = SPACY_MODEL_NAMES
logger = logging.getLogger(__name__)

_UNAVAILABLE_LOG_INTERVAL_SECONDS = 60.0
_FAILED = object()


class SpacyManager:
    def __init__(self):
        self.keep_models_in_memory = config.spacy.keep_models_in_memory
        self.load_retry_ttl_seconds = config.spacy.load_retry_ttl_seconds
        self._cache = {}
        self._failures = {}
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
        return OSError(self._failures[lang][0])

    def _failure_still_active(self, lang: str) -> bool:
        ttl = self.load_retry_ttl_seconds
        if ttl <= 0:
            return False
        failure = self._failures.get(lang)
        if failure is None:
            return True
        return (time.monotonic() - failure[1]) < ttl

    def _clear_failure(self, lang: str) -> None:
        self._cache.pop(lang, None)
        self._failures.pop(lang, None)

    def _usable_cached(self, lang: str, *, allow_expire: bool):
        cached = self._cache.get(lang)
        if cached is _FAILED:
            if self._failure_still_active(lang):
                raise self._failed_error(lang)
            if allow_expire:
                self._clear_failure(lang)
            return None
        if self.keep_models_in_memory and cached is not None:
            return cached
        return None

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
        lang = resolve_spacy_lang(lang)
        cached = self._usable_cached(lang, allow_expire=False)
        if cached is not None:
            return cached
        with self._lock_for(lang):
            cached = self._usable_cached(lang, allow_expire=True)
            if cached is not None:
                return cached
            model_name = MODEL_NAMES.get(lang)
            if model_name is None:
                return self._load_blank(lang)
            return self._load_model(lang, model_name)

    def _load_blank(self, lang: str):
        try:
            nlp = spacy.blank(lang)
        except (ImportError, OSError, ValueError):
            nlp = spacy.blank("xx")
        if self.keep_models_in_memory:
            self._cache[lang] = nlp
        return nlp

    def _load_model(self, lang: str, model_name: str):
        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            message = (
                f"SpaCy model {model_name!r} is not installed. "
                f"NER is disabled for lang={lang}."
            )
            self._failures[lang] = (message, time.monotonic())
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
        self._failures.pop(lang, None)

    def unload_all(self):
        self._cache.clear()
        self._failures.clear()


_spacy_manager = SpacyManager()


def warmup_default_model(lang: str = "en") -> None:
    """Load the default NER pipeline and language detector in background threads."""
    _spacy_manager.warmup(lang)

    def _run_detector() -> None:
        try:
            from src.utils.nlp.lang_detect import (  # pylint: disable=import-outside-toplevel
                warmup_language_detector,
            )

            warmup_language_detector()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    threading.Thread(
        target=_run_detector,
        name="lang-detect-warmup",
        daemon=True,
    ).start()
