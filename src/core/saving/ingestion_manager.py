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
        """
        Initialize the ingestion manager with adapter instances and prepare internal caches.
        
        Parameters:
            embeddings_adapter (EmbeddingsAdapter): Adapter used to produce vector embeddings from text.
            vector_store_adapter (VectorStoreAdapter): Adapter used to persist vectors and return vector IDs.
            graph_adapter (GraphAdapter): Adapter used to interact with the knowledge graph.
        
        Detailed behavior:
            Stores the provided adapters on the instance and initializes `resolved_cache` and `metadata` as empty dictionaries for tracking processed entities and their metadata.
        """
        self.embeddings = embeddings_adapter
        self.vector_store = vector_store_adapter
        self.kg = graph_adapter
        self.resolved_cache = {}
        self.metadata = {}

    def process_node_vectors(self, node_data: ScoutEntity, brain_id):
        """
        Process and store an embedding for the given node and cache its vector id.
        
        If the node's name is already resolved in the manager cache, returns immediately.
        Otherwise embeds the node's name, attaches labels/name/uuid metadata to the embedding,
        stores the resulting vector in the "nodes" vector store under the provided brain, and
        updates the node's properties and the manager's resolved cache with the new vector id.
        
        Parameters:
            node_data (ScoutEntity): Node object whose name will be embedded and whose properties may be updated.
            brain_id: Identifier of the brain/namespace to use when storing vectors.
        
        Returns:
            The node's UUID.
        """
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
        """
        Embed a relationship's description and store the resulting vector in the relationships vector store.
        
        If the relationship has a non-empty description, the description text is embedded and the resulting vector (with metadata including UUID, related node IDs, and predicate) is added to the "relationships" vector store using the provided brain_id. On success the relationship's properties are updated with the stored vector ID.
        
        Parameters:
            rel_data (ArchitectAgentRelationship): Relationship object whose description will be embedded.
            brain_id: Identifier for the brain/context to use when storing the vector.
        
        Returns:
            tuple: (relationship UUID, vector ID string if a vector was stored, otherwise None)
        
        Raises:
            TypeError: If rel_data is not an instance of ArchitectAgentRelationship.
        
        Side effects:
            Updates rel_data.properties['v_id'] when a vector is stored and calls the vector store adapter to persist the embedding.
        """
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