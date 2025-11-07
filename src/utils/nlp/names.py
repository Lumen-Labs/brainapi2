"""
File: /names.py
Created Date: Monday October 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday October 27th 2025 8:20:50 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import Levenshtein


def levenshtein_similarity(name1: str, name2: str) -> float:
    """
    Measure the similarity between two names using the Levenshtein distance.
    """
    return Levenshtein.ratio(name1, name2)


def labels_similarity(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Calculate the similarity between two lists of labels using Jaccard similarity.
    """
    set_a = set(labels_a)
    set_b = set(labels_b)

    if not set_a and not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def find_most_similar_names(
    name: str, names: list[str], labels_a: list[str], labels_list: list[list[str]]
) -> tuple[str, float]:
    """
    Find the most similar name to the given name.
    """
    most_similar_name = None
    max_similarity = 0

    for n, labels in zip(names, labels_list):
        name_sim = levenshtein_similarity(name, n)
        labels_sim = labels_similarity(labels_a, labels)
        similarity = (name_sim * 0.6) + (labels_sim * 0.4)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_name = n

    return most_similar_name, max_similarity


def most_similar_name_with_labels_or_none(
    name: str, names: list[str], labels_a: list[str], labels_list: list[list[str]]
) -> str:
    """
    Find the most similar name to the given name.
    """
    most_similar_name, max_similarity = find_most_similar_names(
        name, names, labels_a, labels_list
    )
    if max_similarity > 0.75:
        return most_similar_name
    return None
