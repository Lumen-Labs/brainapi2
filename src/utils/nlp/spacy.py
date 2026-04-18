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

import subprocess
import sys

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
        except OSError:
            proc = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise OSError(
                    f"SpaCy model {model_name!r} is not installed. "
                    f"Install with: python -m spacy download {model_name}"
                    + (f" ({detail})" if detail else "")
                ) from None
            nlp = spacy.load(model_name)
        if self.keep_models_in_memory:
            self._cache[lang] = nlp
        return nlp

    def unload(self, lang: str):
        if lang in self._cache:
            del self._cache[lang]

    def unload_all(self):
        self._cache.clear()


_spacy_manager = SpacyManager()
