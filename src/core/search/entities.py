"""
File: /entities.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
from src.constants.kg import SearchEntitiesResult
from src.services.kg_agent.main import graph_adapter


import re
import unicodedata


def extract_str_entities_from_text(text: str) -> List[str]:
    """
    Decompose a text into string entity elements.
    """
    if not text or not text.strip():
        return []

    text = unicodedata.normalize("NFKC", text)

    multilingual_stop_words = {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "by",
        "for",
        "to",
        "of",
        "and",
        "or",
        "but",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "his",
        "her",
        "its",
        "our",
        "their",
        "me",
        "him",
        "us",
        "them",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "eines",
        "einem",
        "einen",
        "und",
        "oder",
        "aber",
        "ist",
        "sind",
        "war",
        "waren",
        "wird",
        "wurde",
        "wurden",
        "hat",
        "haben",
        "hatte",
        "hatten",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "de la",
        "et",
        "ou",
        "mais",
        "est",
        "sont",
        "était",
        "étaient",
        "sera",
        "seront",
        "a",
        "ont",
        "avait",
        "avaient",
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "o",
        "pero",
        "es",
        "son",
        "era",
        "eran",
        "será",
        "serán",
        "tiene",
        "tienen",
        "tenía",
        "tenían",
        "il",
        "lo",
        "gli",
        "le",
        "un",
        "una",
        "uno",
        "e",
        "o",
        "ma",
        "è",
        "sono",
        "era",
        "erano",
        "sarà",
        "saranno",
        "ha",
        "hanno",
        "aveva",
        "avevano",
        "de",
        "het",
        "een",
        "en",
        "of",
        "maar",
        "is",
        "zijn",
        "was",
        "waren",
        "zal",
        "zullen",
        "heeft",
        "hebben",
        "had",
        "hadden",
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "e",
        "ou",
        "mas",
        "é",
        "são",
        "era",
        "eram",
        "será",
        "serão",
        "tem",
        "têm",
        "tinha",
        "tinham",
    }

    words_pattern = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)

    words = words_pattern.findall(text)

    entities = []
    seen = set()

    for word in words:
        word_lower = word.lower()

        if word_lower in multilingual_stop_words:
            continue

        if len(word_lower) < 2:
            continue

        if word_lower in seen:
            continue

        if not any(c.isalpha() for c in word):
            continue

        seen.add(word_lower)

        if word and word[0].isupper():
            entities.append(word)
        else:
            entities.append(word.capitalize())

    return entities


def search_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
) -> SearchEntitiesResult:
    """
    Search entities in the knowledge graph.

    Parameters:
        limit (int): Maximum number of entities to return.
        skip (int): Number of entities to skip (offset) for pagination.
        node_labels (Optional[list[str]]): Filter results to nodes matching any of these labels.
        query_text (Optional[str]): Text to match against entity properties or content.
        brain_id (str): Identifier of the graph/brain to query.

    Returns:
        SearchEntitiesResult: Result object containing matched entities and pagination metadata.
    """
    result = graph_adapter.search_entities(
        brain_id, limit, skip, node_labels, query_text
    )
    return result
