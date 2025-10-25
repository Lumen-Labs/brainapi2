"""
File: /KGAgentAddTripletsTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:27:41 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.constants.kg import Node, Predicate, Triple
from src.services.api.constants.tool_schemas import TRIPLE_SCHEMA


class KGAgentAddTripletsTool(BaseTool):
    """
    Tool for adding triplets to the knowledge graph.
    """

    name: str = "kg_agent_add_triplets"
    kg_agent: object
    kg: GraphAdapter
    vector_store: VectorStoreAdapter
    embeddings: EmbeddingsAdapter
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None

    args_schema: dict = {
        "type": "object",
        "properties": {
            "triplets": {
                "type": "array",
                "description": ("The triplets to add to the knowledge graph."),
                "items": TRIPLE_SCHEMA,
            },
        },
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        description: str = (
            "This tool is used to register triplets into the knowledge graph. "
            "Returns the triplets that were added to the knowledge graph. "
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            vector_store=vector_store,
            embeddings=embeddings,
            description=description,
            identification_params=identification_params or {},
            metadata=metadata or {},
        )

    def _run(self, *args, **kwargs) -> str:
        triplets: List[Triple] = []
        for triplet_data in kwargs.get("triplets", []):
            subject = Node(**triplet_data["subject"])
            object_node = Node(**triplet_data["object"])
            predicate = Predicate(
                name=triplet_data["predicate"]["name"].replace(" ", "_").upper(),
                description=triplet_data["predicate"]["description"],
            )

            triplet = Triple(subject=subject, predicate=predicate, object=object_node)
            triplets.append(triplet)

        for triplet in triplets:
            vector = self.embeddings.embed_text(triplet.predicate.description)
            vector.metadata = {
                **(self.metadata or {}),
                "node_ids": [triplet.subject.uuid, triplet.object.uuid],
                "predicate": triplet.predicate.name,
            }
            self.vector_store.add_vectors([vector], "triplets")
            self.kg.add_nodes(
                [triplet.subject, triplet.object],
                self.identification_params,
                self.metadata,
            )
            self.kg.add_relationship(
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
        return f"Triplets added successfully: {triplets}"
