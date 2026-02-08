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

try:
    from spacy.cli import download as spacy_download
except ImportError:
    spacy_download = None

MODEL_NAMES = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "zh": "zh_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "ar": "ar_core_news_sm",
    "hi": "hi_core_news_sm",
    "bn": "bn_core_news_sm",
    "pa": "pa_core_news_sm",
    "ta": "ta_core_news_sm",
    "te": "te_core_news_sm",
    "ml": "ml_core_news_sm",
    "tr": "tr_core_news_sm",
    "vi": "vi_core_news_sm",
    "id": "id_core_news_sm",
    "ms": "ms_core_news_sm",
    "fil": "fil_core_news_sm",
    "th": "th_core_news_sm",
    "lo": "lo_core_news_sm",
    "my": "my_core_news_sm",
    "km": "km_core_news_sm",
}


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
            if spacy_download is None:
                raise
            spacy_download(model_name)
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
