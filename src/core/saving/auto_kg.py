"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 2:29:01 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.core.agents.janitor_agent import JanitorAgentInputOutput
from src.core.agents.scout_agent import ScoutEntity
from src.core.agents.architect_agent import ArchitectAgentRelationship
from src.services.input.agents import (
    embeddings_adapter,
    graph_adapter,
    janitor_agent,
    scout_agent,
    architect_agent,
    vector_store_adapter,
)


class IngestionManager:
    def __init__(
        self,
        embeddings_adapter: EmbeddingsAdapter,
        vector_store_adapter: VectorStoreAdapter,
        graph_adapter: GraphAdapter,
    ):
        self.embeddings = embeddings_adapter
        self.vector_store = vector_store_adapter
        self.kg = graph_adapter
        self.resolved_cache = {}

    def process_node_vectors(self, node_data: ScoutEntity, brain_id):
        if node_data.name in self.resolved_cache:
            return node_data.uuid

        v_sub = self.embeddings.embed_text(node_data.name)
        v_sub.metadata = {
            "labels": [node_data.type],
            "name": node_data.name,
            "uuid": node_data.uuid,
        }
        if v_sub.embeddings:
            v_ids = self.vector_store.add_vectors(
                [v_sub], store="nodes", brain_id=brain_id
            )
            node_data.properties = {
                **(node_data.properties or {}),
                "v_id": v_ids[0],
            }
            self.resolved_cache[node_data.name] = v_ids[0]
            return node_data.uuid
        else:
            print("[ ! ]Node not embedded:", node_data)
        return node_data.uuid

    def process_rel_vectors(self, rel_data: ArchitectAgentRelationship, brain_id):
        if hasattr(rel_data, "description") and rel_data.description:
            v_rel = self.embeddings.embed_text(rel_data.description)
            v_rel.metadata = {
                **(self.metadata or {}),
                "uuid": rel_data.uuid,
                "node_ids": [rel_data.tail.uuid, rel_data.tip.uuid],
                "predicate": rel_data.name,
            }
            if v_rel.embeddings:
                v_ids = self.vector_store.add_vectors(
                    [v_rel], store="relationships", brain_id=brain_id
                )
                rel_data.properties = {
                    **(rel_data.properties or {}),
                    "v_id": v_ids[0],
                }
            else:
                print("[ ! ]Relationship not embedded:", rel_data)
        return rel_data.uuid


def enrich_kg_from_input(
    input: str, targeting: Optional[Node] = None, brain_id: str = "default"
) -> str:
    """
    Enrich the knowledge graph from the input.
    """

    input_tokens = 0
    output_tokens = 0

    manager = IngestionManager(embeddings_adapter, vector_store_adapter, graph_adapter)

    entities = scout_agent.run(input, targeting=targeting, brain_id=brain_id)
    input_tokens += entities.input_tokens
    output_tokens += entities.output_tokens
    # logtable(
    #     [entity.model_dump(mode="json") for entity in entities.entities],
    #     title="Entities",
    # )

    architect_response = architect_agent.run(
        input, entities.entities, targeting=targeting, brain_id=brain_id
    )
    input_tokens += architect_response.input_tokens
    output_tokens += architect_response.output_tokens
    # logtable(
    #     [new_node.model_dump(mode="json") for new_node in architect_response.new_nodes],
    #     title="New Nodes",
    # )
    # logtable(
    #     [
    #         relationship.model_dump(mode="json")
    #         for relationship in architect_response.relationships
    #     ],
    #     title="Relationships",
    # )

    normalized_entities = []
    normalized_virtual_nodes = []
    normalized_relationships = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        entity_futures = [
            executor.submit(
                janitor_agent.run,
                JanitorAgentInputOutput(entity=entity),
                text=input,
                targeting=targeting,
                brain_id=brain_id,
            )
            for entity in entities.entities
        ]

        virtual_node_futures = [
            executor.submit(
                janitor_agent.run,
                JanitorAgentInputOutput(virtual_node=new_node),
                text=input,
                targeting=targeting,
                brain_id=brain_id,
            )
            for new_node in architect_response.new_nodes
        ]

        all_futures = list(entity_futures) + list(virtual_node_futures)

        for future in all_futures:
            try:
                normalized_result = future.result(timeout=180)
                input_tokens += normalized_result.input_tokens or 0
                output_tokens += normalized_result.output_tokens or 0
                if normalized_result.entity:
                    normalized_entities.append(normalized_result.entity)
                elif normalized_result.virtual_node:
                    normalized_virtual_nodes.append(normalized_result.virtual_node)
                else:
                    print("Entity/Virtual node not normalized:", normalized_result)
            except FutureTimeoutError:
                print(
                    "[!] Janitor agent entity/virtual_node future timed out, skipping"
                )
                continue
            except Exception as e:
                print(f"[!] Janitor agent entity/virtual_node future failed: {e}")
                continue

        relationship_futures = [
            executor.submit(
                janitor_agent.run,
                JanitorAgentInputOutput(relationship=relationship),
                text=input,
                targeting=targeting,
                brain_id=brain_id,
            )
            for relationship in architect_response.relationships
        ]

        for future in relationship_futures:
            try:
                normalized_result = future.result(timeout=180)
                input_tokens += normalized_result.input_tokens or 0
                output_tokens += normalized_result.output_tokens or 0
                if normalized_result.relationship:
                    normalized_relationships.append(normalized_result.relationship)
                else:
                    print("Relationship not normalized:", normalized_result)
            except FutureTimeoutError:
                print("[!] Janitor agent relationship future timed out, skipping")
                continue
            except Exception as e:
                print(f"[!] Janitor agent relationship future failed: {e}")
                continue

    graph_nodes = []

    with ThreadPoolExecutor(max_workers=10) as io_executor:
        node_embedding_futures = []

        for entity in normalized_entities:
            future = io_executor.submit(manager.process_node_vectors, entity, brain_id)
            node_embedding_futures.append((future, entity))

        for virtual_node in normalized_virtual_nodes:
            future = io_executor.submit(
                manager.process_node_vectors, virtual_node, brain_id
            )
            node_embedding_futures.append((future, virtual_node))

        for future, node_data in node_embedding_futures:
            try:
                future.result(timeout=180)
                graph_nodes.append(
                    Node(
                        uuid=node_data.uuid,
                        labels=[node_data.type],
                        name=node_data.name,
                        description=node_data.description,
                        properties=node_data.properties,
                    )
                )
            except FutureTimeoutError:
                print(
                    f"[!] Node embedding future timed out for {node_data.name}, skipping"
                )
                continue
            except Exception as e:
                print(f"[!] Node embedding future failed for {node_data.name}: {e}")
                continue

        graph_adapter.add_nodes(graph_nodes, brain_id=brain_id)

        for relationship in normalized_relationships:
            io_executor.submit(manager.process_rel_vectors, relationship, brain_id)
            graph_adapter.add_relationship(
                Node(
                    uuid=relationship.tail.uuid,
                    labels=[relationship.tail.type],
                    name=relationship.tail.name,
                ),
                Predicate(
                    uuid=relationship.uuid,
                    name=relationship.name,
                    description=relationship.description,
                    properties=relationship.properties,
                ),
                Node(
                    uuid=relationship.tip.uuid,
                    labels=[relationship.tip.type],
                    name=relationship.tip.name,
                ),
                brain_id=brain_id,
            )

    print("-----------------------------------")
    print("> Input tokens:   ", input_tokens)
    print("> Output tokens:  ", output_tokens)
    print("-----------------------------------")
