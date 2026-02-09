"""
File: /lang_detect.py
Project: nlp
Created Date: Sunday February 8th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 8th 2026 11:21:40 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

import functools
from langid import classify

from src.utils.nlp.spacy import MODEL_NAMES, _spacy_manager


@functools.lru_cache(maxsize=2048)
def _langid_cached(text: str) -> tuple[str, float]:
    return classify(text)


def get_lang_detector(nlp, name):
    return LanguageDetector()


nlp_det = spacy.blank("en")
Language.factory("language_detector", func=get_lang_detector)
nlp_det.add_pipe("language_detector", last=True)


def detect_language(text: str) -> str:
    return nlp_det(text)._.language["language"]


def get_nlp_for_text(text: str):
    lang = detect_language(text)
    if lang not in MODEL_NAMES:
        return None
    return _spacy_manager.get_model(lang)


def langid_detect(text: str) -> tuple[str, float]:
    return _langid_cached(text)
