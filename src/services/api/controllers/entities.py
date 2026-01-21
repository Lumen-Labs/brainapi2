"""
File: /entities.py
Project: controllers
Created Date: Sunday January 18th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday January 18th 2026 9:38:31 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Literal
from src.core.search.entity_context import EntityContext
from src.core.search.entity_info import EventSynergyRetriever
from src.services.api.constants.requests import (
    GetEntityInfoResponse,
    GetEntityContextResponse,
    GetEntitySibilingsResponse,
)
from src.core.search.entity_sibilings import EntitySinergyRetriever


async def get_entity_info(
    target: str, query: str, max_depth: int = 3, brain_id: str = "default"
) -> GetEntityInfoResponse:
    """
    Get the entity info for a given query.
    """
    event_synergy_retriever = EventSynergyRetriever(brain_id)
    paths = event_synergy_retriever.retrieve_matches(target, query, max_depth)

    return GetEntityInfoResponse(target_node=paths.target_node, path=paths)


async def get_entity_context(
    target: str, context_depth: int = 3, brain_id: str = "default"
) -> GetEntityContextResponse:
    """
    Get the entity context for a given target.
    """
    entity_context = EntityContext(target, brain_id)
    target_node, neighborhood, text_contexts, natural_language_web = (
        entity_context.get_context(context_depth=context_depth)
    )
    return GetEntityContextResponse(
        target_node=target_node,
        neighborhood=neighborhood,
        text_contexts=text_contexts,
        natural_language_web=natural_language_web,
    )


async def get_entity_sibilings(
    target: str,
    polarity: Literal["same", "opposite"] = "same",
    brain_id: str = "default",
) -> GetEntitySibilingsResponse:
    """
    Get the entity siblings for a given target.
    """
    entity_sibilings = EntitySinergyRetriever(brain_id)
    target_node, synergies = entity_sibilings.retrieve_sibilings(target, polarity)
    return GetEntitySibilingsResponse(
        target_node=target_node,
        synergies=synergies,
    )
