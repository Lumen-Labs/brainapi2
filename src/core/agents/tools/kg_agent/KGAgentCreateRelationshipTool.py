"""
File: /KGAgentCreateRelationshipTool.py
Project: kg_agent
Created Date: Saturday January 31st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 31st 2026 11:17:56 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import uuid
from langchain.tools import BaseTool

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate


class KGAgentCreateRelationshipTool(BaseTool):
    """
    Tool for creating a relationship in the knowledge graph.
    """

    name: str = "kg_agent_create_relationship"

    kg_agent: object
    kg: GraphAdapter
    brain_id: str = "default"
    embeddings_adapter: EmbeddingsAdapter
    vector_store_adapter: VectorStoreAdapter

    args_schema: dict = {
        "type": "object",
        "properties": {
            "tail": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the tail node.",
                    },
                    "labels": {
                        "type": "array",
                        "description": "The labels of the tail node.",
                        "items": {
                            "type": "string",
                        },
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the tail node.",
                    },
                    "polarity": {
                        "type": "string",
                        "description": "The polarity of the node, describing if the node rapresents a good thing or bad.",
                        "enum": ["positive", "negative", "neutral"],
                    },
                },
                "required": ["name", "labels"],
            },
            "predicate": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the predicate.",
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the predicate.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "The amount of the predicate.",
                    },
                },
                "required": ["name", "description"],
            },
            "head": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the tail node.",
                    },
                    "labels": {
                        "type": "array",
                        "description": "The labels of the tail node.",
                        "items": {
                            "type": "string",
                        },
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the tail node.",
                    },
                    "polarity": {
                        "type": "string",
                        "description": "The polarity of the node, describing if the node rapresents a good thing or bad.",
                        "enum": ["positive", "negative", "neutral"],
                    },
                },
                "required": ["name", "labels"],
            },
        },
        "required": ["tail", "predicate", "head"],
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        embeddings_adapter: EmbeddingsAdapter,
        vector_store_adapter: VectorStoreAdapter,
        brain_id: str = "default",
    ):
        description: str = (
            "Use this tool to create a relationship in the knowledge graph. "
            "The argument should be a valid JSON object with a 'tail', 'predicate', and 'head' keys. "
            "The 'tail' key should be the tail node of the relationship. "
            "The 'predicate' key should be the predicate of the relationship. "
            "The 'head' key should be the head node of the relationship. "
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            embeddings_adapter=embeddings_adapter,
            vector_store_adapter=vector_store_adapter,
            description=description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Create a node in the knowledge graph.
        """
        _tail: Node = None
        _predicate: Predicate = None
        _head: Node = None

        flow_key = str(uuid.uuid4())

        if len(kwargs) > 0:
            args_tail = kwargs.get("tail", {})
            if isinstance(args_tail, dict):
                _tail = Node(**args_tail)
            args_predicate = kwargs.get("predicate", {})
            if isinstance(args_predicate, dict):
                if not args_predicate.get("name"):
                    return "Error: predicate name is required"
                if not args_predicate.get("description"):
                    return "Error: predicate description is required"
                _predicate = Predicate(**args_predicate)
            args_head = kwargs.get("head", {})
            if isinstance(args_head, dict):
                _head = Node(**args_head)

        if _tail is None or _predicate is None or _head is None:
            return "Error: tail, predicate, and head are required"

        embedding_v = self.embeddings_adapter.embed_text(_predicate.description)
        vs = self.vector_store_adapter.add_vectors(
            [embedding_v], store="relationships", brain_id=self.brain_id
        )
        _predicate.properties = {
            **(_predicate.properties or {}),
            "v_id": vs[0],
        }
        _predicate.flow_key = flow_key

        self.kg.add_relationship(_tail, _predicate, _head, brain_id=self.brain_id)

        return "Relationship created successfully"
