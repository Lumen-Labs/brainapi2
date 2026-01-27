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

from typing import List, Literal
from src.core.search.entity_context import EntityContext
from src.core.search.entity_info import EventSynergyRetriever
from src.services.api.constants.requests import (
    GetEntityInfoResponse,
    GetEntityContextResponse,
    GetEntitySibilingsResponse,
    GetEntityStatusResponse,
)
from src.core.search.entity_sibilings import EntitySinergyRetriever
from src.services.data.main import data_adapter
from src.services.input.agents import embeddings_adapter
from src.services.kg_agent.main import graph_adapter, vector_store_adapter


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


async def get_entity_status(
    target: str,
    types: List[str] = [],
    brain_id: str = "default",
) -> GetEntityStatusResponse:
    """
    Get the entity status for a given target.
    """

    target_embeddings = embeddings_adapter.embed_text(target)
    target_node_vs = vector_store_adapter.search_vectors(
        target_embeddings.embeddings, store="nodes", brain_id=brain_id
    )

    target_node = None

    for target_node_v in target_node_vs:
        target_node_id = target_node_v.metadata.get("uuid")
        target_node = graph_adapter.get_by_uuids([target_node_id], brain_id=brain_id)[0]

        if len(types) > 0:
            if set(target_node.labels).intersection(set(types)):
                break
        else:
            break

    if not target_node:
        return GetEntityStatusResponse(
            node=None,
            exists=False,
            has_relationships=False,
            relationships=[],
            observations=[],
        )

    rel_tuples = graph_adapter.get_neighbors([target_node], brain_id=brain_id)

    observations = data_adapter.get_observations_list(
        brain_id=brain_id, resource_id=target_node.uuid
    )

    return GetEntityStatusResponse(
        node=target_node,
        exists=True,
        has_relationships=len(rel_tuples[target_node.uuid]) > 0,
        relationships=rel_tuples[target_node.uuid],
        observations=observations,
    )
