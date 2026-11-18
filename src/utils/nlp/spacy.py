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

import spacy
from src.config import config
from src.constants.spacy_models import SPACY_MODEL_NAMES

MODEL_NAMES = SPACY_MODEL_NAMES


class SpacyManager:
    def __init__(self):
        self.keep_models_in_memory = config.spacy.keep_models_in_memory
        self._cache = {}

    def get_model(self, lang: str):
        if lang not in MODEL_NAMES:
            raise ValueError(f"Unsupported language: {lang}")
        if self.keep_models_in_memory and lang in self._cache:
            return self._cache[lang]
        model_name = MODEL_NAMES[lang]
        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise OSError(
                f"SpaCy model {model_name!r} is not installed. "
                f"Install with: python -m spacy download {model_name}"
            ) from exc
        if self.keep_models_in_memory:
            self._cache[lang] = nlp
        return nlp

    def unload(self, lang: str):
        if lang in self._cache:
            del self._cache[lang]

    def unload_all(self):
        self._cache.clear()


_spacy_manager = SpacyManager()
