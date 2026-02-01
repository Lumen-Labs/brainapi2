"""
File: /entities.py
Project: controllers
Created Date: Sunday January 18th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
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
    Retrieve matching event paths for a target and query up to a specified traversal depth.

    Parameters:
        target (str): The entity identifier or text to locate.
        query (str): The query text used to find relevant event matches.
        max_depth (int): Maximum path traversal depth to consider when retrieving matches.
        brain_id (str): Brain/workspace identifier to scope the retrieval.

    Returns:
        GetEntityInfoResponse: Contains the located target node (`target_node`) and the retrieved paths (`path`).
    """
    event_synergy_retriever = EventSynergyRetriever(brain_id)
    paths = event_synergy_retriever.retrieve_matches(target, query, max_depth)

    return GetEntityInfoResponse(target_node=paths.target_node, path=paths)


async def get_entity_context(
    target: str, context_depth: int = 3, brain_id: str = "default"
) -> GetEntityContextResponse:
    """
    Retrieve contextual information for the specified entity target.

    Parameters:
        target (str): The entity identifier or text to retrieve context for.
        context_depth (int): Maximum graph depth (number of hops) to include in the neighborhood.
        brain_id (str): Identifier of the brain/workspace to query.

    Returns:
        GetEntityContextResponse: Response with the following fields:
            target_node: The node representing the target entity.
            neighborhood: Nearby nodes and relationships up to `context_depth`.
            text_contexts: Relevant text excerpts or documents associated with the target and neighborhood.
            natural_language_web: A natural-language representation or summary of the surrounding context.
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
    Retrieve sibling entities (synergies) for a target entity.

    Parameters:
        polarity (Literal["same", "opposite"]): Which polarity of siblings to return — "same" for similar entities, "opposite" for contrasted entities.

    Returns:
        GetEntitySibilingsResponse: Object containing the resolved target node and its list of synergies.
    """
    entity_sibilings = EntitySinergyRetriever(brain_id)
    target_node, synergies, seed_nodes, potential_anchors = (
        entity_sibilings.retrieve_sibilings(target, polarity)
    )
    return GetEntitySibilingsResponse(
        target_node=target_node,
        synergies=synergies,
        anchors=seed_nodes,
        potential_anchors=potential_anchors,
    )


async def get_entity_status(
    target: str,
    types: List[str] = [],
    brain_id: str = "default",
) -> GetEntityStatusResponse:
    """
    Retrieve status information for an entity matching the provided target text.

    Parameters:
        target (str): Text used to locate the entity.
        types (List[str]): Optional list of node label types to filter matches; if provided, the first matching node whose labels intersect `types` is chosen.
        brain_id (str): Identifier of the brain/workspace to query.

    Returns:
        GetEntityStatusResponse: Response containing the matched node (or `None` if not found), `exists` indicating presence, `has_relationships` indicating whether the node has neighbors, `relationships` listing neighbor tuples, and `observations` associated with the node. When no matching node is found, `exists` is `False` and `relationships` and `observations` are empty.
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
