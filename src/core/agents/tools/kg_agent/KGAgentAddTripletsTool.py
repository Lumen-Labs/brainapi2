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
from uuid import uuid4
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.constants.kg import Node, Predicate, Triple
from src.services.api.constants.tool_schemas import TRIPLE_SCHEMA
from src.utils.nlp.names import most_similar_name_with_labels_or_none


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
            try:
                subject = Node(**triplet_data["subject"])
                object_node = Node(**triplet_data["object"])
            except Exception as e:
                print(f"Error creating nodes: {e} -  {triplet_data}")
                continue

            v_sub = self.embeddings.embed_text(subject.name)
            v_obj = self.embeddings.embed_text(object_node.name)

            v_sim_sub = self.vector_store.search_vectors(v_sub.embeddings, "nodes", k=5)
            v_sim_obj = self.vector_store.search_vectors(v_obj.embeddings, "nodes", k=5)
            print("[v_sim_x]", v_sim_sub, v_sim_obj)
            sim_sub = most_similar_name_with_labels_or_none(
                subject.name,
                [v.metadata.get("name", []) for v in v_sim_sub],
                subject.labels,
                [v.metadata.get("labels", []) for v in v_sim_sub],
            )
            sim_obj = most_similar_name_with_labels_or_none(
                object_node.name,
                [v.metadata.get("name", []) for v in v_sim_obj],
                object_node.labels,
                [v.metadata.get("labels", []) for v in v_sim_obj],
            )
            print("[sim_x]", sim_sub, sim_obj)
            if sim_sub:
                sim_sub_vector = next(
                    (v for v in v_sim_sub if v.metadata.get("name", []) == sim_sub),
                    None,
                )
                print("[sim_sub_vector]", sim_sub_vector)
                subject = Node(
                    name=sim_sub_vector.metadata.get("name", []),
                    uuid=sim_sub_vector.metadata.get("uuid"),
                    labels=sim_sub_vector.metadata.get("labels", []),
                )
                # TODO: update changelog to record any eventual merge if sim_sub is not the same as subject.name
            else:
                v_sub.metadata = {
                    "name": subject.name,
                    "labels": subject.labels,
                    "uuid": str(uuid4()),
                }
                vector = self.vector_store.add_vectors([v_sub], store="nodes")
                subject.uuid = v_sub.metadata.get("uuid")

            if sim_obj:
                sim_obj_vector = next(
                    (v for v in v_sim_obj if v.metadata.get("name", []) == sim_obj),
                    None,
                )
                print("[sim_obj_vector]", sim_obj_vector)
                object_node = Node(
                    name=sim_obj_vector.metadata.get("name", []),
                    uuid=sim_obj_vector.metadata.get("uuid"),
                    labels=sim_obj_vector.metadata.get("labels", []),
                )
                # TODO: update changelog to record any eventual merge if sim_obj is not the same as object_node.name
            else:
                v_obj.metadata = {
                    "name": object_node.name,
                    "labels": object_node.labels,
                    "uuid": str(uuid4()),
                }
                vector = self.vector_store.add_vectors([v_obj], store="nodes")
                object_node.uuid = v_obj.metadata.get("uuid")

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

            # TODO: add changelog relationship created

        return f"Triplets added successfully: {triplets}"
