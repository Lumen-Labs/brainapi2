"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from typing import List, Optional, Tuple

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.config import config
from src.constants.kg import Node, Predicate
from src.core.agents.janitor_agent import JanitorAgent
from src.core.agents.scout_agent import ScoutAgent, ScoutEntity
from src.core.agents.architect_agent import ArchitectAgent, ArchitectAgentRelationship
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.lib.neo4j.client import _neo4j_client
from src.utils.tokens import merge_token_details, token_detail_from_token_counts


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
        self.metadata = {}

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
        if not isinstance(rel_data, ArchitectAgentRelationship):
            raise TypeError(
                f"Expected ArchitectAgentRelationship, got {type(rel_data)}"
            )
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
        return rel_data.uuid, (
            rel_data.properties.get("v_id") if rel_data.properties else None
        )


def enrich_kg_from_input(
    input: str, targeting: Optional[Node] = None, brain_id: str = "default"
) -> str:
    """
    Enrich the knowledge graph from the input.
    """

    scout_agent = ScoutAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
    )
    architect_agent = ArchitectAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
    )

    token_details = []

    manager = IngestionManager(embeddings_adapter, vector_store_adapter, graph_adapter)

    entities = scout_agent.run(input, targeting=targeting, brain_id=brain_id)
    token_details.append(
        token_detail_from_token_counts(
            scout_agent.input_tokens,
            scout_agent.output_tokens,
            scout_agent.cached_tokens,
            scout_agent.reasoning_tokens,
        )
    )

    architect_response = architect_agent.run_tooler(
        input, entities.entities, targeting=targeting, brain_id=brain_id
    )
    token_details.append(
        token_detail_from_token_counts(
            architect_agent.input_tokens,
            architect_agent.output_tokens,
            architect_agent.cached_tokens,
            architect_agent.reasoning_tokens,
        )
    )

    graph_nodes = []

    with ThreadPoolExecutor(max_workers=10) as io_executor:
        node_embedding_futures = []

        for entity in entities.entities:
            future = io_executor.submit(manager.process_node_vectors, entity, brain_id)
            node_embedding_futures.append((future, entity))

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

        relationships_list = architect_response
        flattened_relationships = []
        for item in relationships_list:
            if isinstance(item, list):
                flattened_relationships.extend(item)
            else:
                flattened_relationships.append(item)

        rel_embedding_futures: List[Tuple[Future, ArchitectAgentRelationship]] = []
        for relationship in flattened_relationships:
            if not isinstance(relationship, ArchitectAgentRelationship):
                print(f"[!] Skipping invalid relationship type: {type(relationship)}")
                continue
            future = io_executor.submit(
                manager.process_rel_vectors, relationship, brain_id
            )
            rel_embedding_futures.append((future, relationship))

        for future, relationship in rel_embedding_futures:
            try:
                v_id, v_rel_id = future.result(timeout=180)
                graph_adapter.add_relationship(
                    Node(
                        uuid=relationship.tail.uuid,
                        flow_key=relationship.tail.flow_key,
                        labels=[relationship.tail.type],
                        name=relationship.tail.name,
                        **(
                            {"happened_at": relationship.tail.happened_at}
                            if relationship.tail.happened_at
                            else {}
                        ),
                    ),
                    Predicate(
                        uuid=relationship.uuid,
                        flow_key=relationship.flow_key,
                        name=relationship.name,
                        description=relationship.description,
                        properties={
                            **(relationship.properties or {}),
                            "v_id": v_rel_id,
                        },
                    ),
                    Node(
                        uuid=relationship.tip.uuid,
                        flow_key=relationship.tip.flow_key,
                        labels=[relationship.tip.type],
                        name=relationship.tip.name,
                        **(
                            {"happened_at": relationship.tip.happened_at}
                            if relationship.tip.happened_at
                            else {}
                        ),
                    ),
                    brain_id=brain_id,
                )
            except FutureTimeoutError:
                rel_name = getattr(relationship, "name", "unknown")
                print(
                    f"[!] Relationship embedding future timed out for {rel_name}, skipping"
                )
                continue
            except Exception as e:
                rel_name = getattr(relationship, "name", "unknown")
                print(f"[!] Relationship embedding future failed for {rel_name}: {e}")
                continue

    token_details = merge_token_details(
        [scout_agent.token_detail, architect_agent.token_detail]
    )

    print("-----------------------------------")
    print(
        f"> Input tokens:   {token_details.input.total} -> ${token_details.input.total * config.pricing.input_token_price}"
    )
    print(
        f"> Output tokens:  {token_details.output.total} -> ${token_details.output.total * config.pricing.output_token_price}"
    )
    print(
        f"> Cost -> ${token_details.input.total * config.pricing.input_token_price + token_details.output.total * config.pricing.output_token_price} for {len(input)} characters"
    )
    print(
        f"> Cost/Character -> ${((token_details.input.total * config.pricing.input_token_price + token_details.output.total * config.pricing.output_token_price) / len(input)):.12f}"
    )
    print("-----------------------------------")
