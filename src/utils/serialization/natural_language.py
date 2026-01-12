"""
File: /natural_language.py
Created Date: Friday January 2nd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday January 2nd 2026 10:59:44 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Tuple


from src.constants.kg import Node, Predicate


def hops_to_natural_language(
    hops: List[Tuple[Node, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]],
) -> str:
    """
    Currently only supports 2nd degree hops.
    """
    pass
