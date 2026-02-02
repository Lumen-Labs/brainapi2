"""
File: /numbers.py
Project: similarity
Created Date: Monday February 2nd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday February 2nd 2026 8:40:10 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List


def wsim(similarity: float, weight: float) -> float:
    """
    Weighten the similarity by the weight.
    eg: similarity = 0.25, weight = 0.25 -> new similarity = 0.25^(1-0.25) = 0.35
    eg: similarity = 0.25, weight = 0.75 -> new similarity = 0.25^(1-0.75) = 0.71

    eg: similarity = 0.25, weight = -0.25 -> new similarity = 0.25^(1+0.25) = 0.18
    eg: similarity = 0.25, weight = -0.75 -> new similarity = 0.25^(1+0.75) = 0.09

    Arguments:
        similarity (float): The similarity to weighten.
        weight (float): 0.0 to 1.0 - The weight to weighten the similarity by.

    Returns:
        float: The weighted similarity.
    """
    return similarity ** (1 - weight)


def wmean(values: List[float], weight: float) -> float:
    """
    Weighted mean of a list of values.
    """

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    n = len(values)
    # Modified: ensure effective_weight is not zero for n=2
    # The original was: weight * max(0, n - 2) / max(2, n)
    effective_weight = weight * max(1, n - 2) / max(2, n)
    transformed = [
        wsim(values[0], effective_weight),
        wsim(values[1], effective_weight),
    ] + list(values[2:])
    return sum(transformed) / len(transformed)
