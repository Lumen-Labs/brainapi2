"""
File: /KGAgentDeleteRelationshipTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:29:22 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional, Tuple
from uuid import uuid4
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.constants.data import (
    KGChangeLogRelationshipDeprecated,
    KGChanges,
    KGChangesType,
)
from src.constants.kg import Node, Predicate
from src.services.data.main import data_adapter


class KGAgentDeleteRelationshipTool(BaseTool):
    """
    Tool for deleting a relationship from the knowledge graph.
    """

    name: str = "kg_agent_delete_relationship"
    kg_agent: object
    kg: GraphAdapter
    vector_store: VectorStoreAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "description": ("The labels of the subject node."),
                        "items": {
                            "type": "string",
                        },
                    },
                    "name": {
                        "type": "string",
                        "description": ("The name of the subject node."),
                    },
                },
                "required": ["labels", "name"],
            },
            "predicate": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": ("The name of the predicate."),
                    },
                },
                "required": ["name"],
            },
            "object": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "description": ("The labels of the object node."),
                        "items": {
                            "type": "string",
                        },
                    },
                    "name": {
                        "type": "string",
                        "description": ("The name of the object node."),
                    },
                },
                "required": ["labels", "name"],
            },
        },
        "required": ["subject", "predicate", "object"],
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        brain_id: str = "default",
    ):
        description: str = (
            "This tool is used to deprecate a relationship that is no longer valid from the knowledge graph. "
            "Returns the relationship that was deprecated in the knowledge graph if it was found else returns an error message."
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            vector_store=vector_store,
            description=description,
            brain_id=brain_id or "default",
        )

    def _run(self, *args, **kwargs) -> str:
        subject = Node(**kwargs.get("subject"))
        predicate = Predicate(**kwargs.get("predicate"))
        object_node = Node(**kwargs.get("object"))

        deprecated_relationship: Tuple[Node, Predicate, Node] | None = (
            self.kg.deprecate_relationship(
                subject,
                predicate,
                object_node,
                brain_id=self.brain_id,
            )
        )
        if deprecated_relationship:
            kg_changes = KGChanges(
                type=KGChangesType.RELATIONSHIP_DEPRECATED,
                change=KGChangeLogRelationshipDeprecated(
                    type=KGChangesType.RELATIONSHIP_DEPRECATED,
                    subject=deprecated_relationship[0],
                    predicate=deprecated_relationship[1],
                    object=deprecated_relationship[2],
                ),
            )
            data_adapter.save_kg_changes(kg_changes, brain_id=self.brain_id)
            return f"Relationship deprecated successfully: {deprecated_relationship}"

        return "Relationship not found in the knowledge graph."
