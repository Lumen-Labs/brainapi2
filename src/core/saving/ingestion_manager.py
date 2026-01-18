"""
File: /ingestion_manager.py
Project: saving
Created Date: Sunday January 18th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday January 18th 2026 4:04:33 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from src.adapters.embeddings import EmbeddingsAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.embeddings import VectorStoreAdapter
from src.core.agents.scout_agent import ScoutEntity
from src.core.agents.architect_agent import ArchitectAgentRelationship


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
