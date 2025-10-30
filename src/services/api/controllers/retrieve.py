"""
File: /retrieve.py
Created Date: Sunday October 26th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 26th 2025 4:03:07 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from typing import Optional

from fastapi import HTTPException

from src.constants.kg import IdentificationParams, Predicate, Relationship
from src.services.api.constants.requests import (
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
)
from src.services.kg_agent.main import graph_adapter, kg_agent
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter


async def retrieve_data(
    text: str, limit: int, preferred_entities: str
) -> RetrieveRequestResponse:
    """
    Retrieve data from the knowledge graph and data store.
    """
    if preferred_entities:
        preferred_entities = [
            e.strip() for e in preferred_entities.split(",") if e.strip()
        ]
    else:
        preferred_entities = []

    def _get_data():
        text_embeddings = embeddings_adapter.embed_text(text)
        data_vectors = vector_store_adapter.search_vectors(
            text_embeddings.embeddings, "data", limit
        )
        triple_vectors = vector_store_adapter.search_vectors(
            text_embeddings.embeddings, "triplets", limit
        )

        search_result = data_adapter.search(text)

        ts_text_chunks = search_result.text_chunks
        ts_observations = search_result.observations

        v_text_chunks, v_observations = data_adapter.get_text_chunks_by_ids(
            [dv.metadata.get("resource_id") for dv in data_vectors], True
        )

        node_ids = [
            node_id
            for tv in triple_vectors
            for node_id in tv.metadata.get("node_ids", [])
        ]
        print("[preferred entities]", preferred_entities, type(preferred_entities))
        nodes = graph_adapter.get_nodes_by_uuid(
            uuids=node_ids,
            with_relationships=True,
            relationships_depth=1,
            relationships_type=[
                tv.metadata.get("predicate")
                for tv in triple_vectors
                if tv.metadata.get("predicate", None)
            ],
            preferred_labels=preferred_entities or [],
        )
        return ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes

    ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes = (
        await asyncio.to_thread(_get_data)
    )

    return RetrieveRequestResponse(
        data=[*ts_text_chunks, *v_text_chunks],
        observations=[*ts_observations, *v_observations],
        relationships=nodes,
    )


async def retrieve_neighbors(
    uuid: Optional[str] = None,
    identification_params: Optional[IdentificationParams] = None,
    limit: int = 10,
) -> RetrieveNeighborsRequestResponse:
    def _get_neighbors():
        node = None
        if uuid:
            node = graph_adapter.get_by_uuid(uuid)
        elif identification_params:
            node = graph_adapter.get_by_identification_params(
                identification_params, entity_types=identification_params.entity_types
            )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        raw = graph_adapter.get_neighbors(node, limit)

        normalized = []
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 2:
                n, pred = item[0], item[1]
            else:
                n, pred = item, None

            if isinstance(pred, Predicate):
                relation = Relationship(
                    direction=pred.direction or "out", predicate=pred
                )
            else:
                relation = Relationship(
                    direction="out",
                    predicate=Predicate(name="RELATED_TO", description=""),
                )

            if not n.uuid == node.uuid:
                # TODO: provide the common node too and maybe also observations
                normalized.append(
                    RetrievedNeighborNode(
                        **n.model_dump(),
                        relation=relation,
                        observations=[],
                    )
                )

        if len(normalized) < limit:
            neighbor_nodes = graph_adapter.get_nodes_by_uuid(
                uuids=[n.uuid for n in normalized], with_relationships=True
            )
            v_neighbors = vector_store_adapter.search_similar_by_ids(
                vector_ids=[n.get("node").uuid for n in neighbor_nodes],
                store="nodes",
                min_similarity=0.5,
                limit=limit - len(normalized),
            )
            print("[v_neighbors]", v_neighbors)
            # TODO: get v_neighbor nodes by ids and the k (the common)
            # TODO + push to normalized
            # TODO (directly the nodes with label same as main, the others search for neighbors with same label of main)

        if len(normalized) < limit:
            v_data = vector_store_adapter.search_similar_by_ids(
                vector_ids=[n.uuid for n in normalized],
                store="data",
                min_similarity=0.5,
                limit=limit - len(normalized),
            )
            print("[v_data]", v_data)
            # TODO: get v_neighbor nodes by ids and the k (the common)
            # TODO + push to normalized
            # TODO (directly the nodes with label same as main, the others search for neighbors with same label of main)

        return RetrieveNeighborsRequestResponse(neighbors=normalized)

    return await asyncio.to_thread(_get_neighbors)


async def retrieve_neighbors_ai_mode(
    identification_params: IdentificationParams,
    looking_for: Optional[list[str]],
    limit: int,
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """

    def _get_neighbors():
        node = graph_adapter.get_by_identification_params(
            identification_params, entity_types=identification_params.entity_types
        )
        print("[node]", node)
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        result = kg_agent.retrieve_neighbors(node, looking_for, limit)
        print("[result]", result)

        ids = [neighbor.uuid for neighbor in result.neighbors]
        descriptions = [neighbor.description for neighbor in result.neighbors]

        nodes = graph_adapter.get_nodes_by_uuid(uuids=ids)
        paired = list(zip(nodes, descriptions))

        return RetrieveNeighborsRequestResponse(neighbors=paired)

    result = await asyncio.to_thread(_get_neighbors)

    return result
