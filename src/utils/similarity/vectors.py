"""
File: /vectors.py
Created Date: Sunday January 4th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday January 4th 2026 12:32:25 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import numpy as np
from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return np.linalg.norm(vec1_np - vec2_np)
