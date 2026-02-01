"""
File: /KGAgentCreateNodeTool.py
Project: kg_agent
Created Date: Saturday January 31st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 31st 2026 11:18:17 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node


class KGAgentCreateNodeTool(BaseTool):
    """
    Tool for creating a node in the knowledge graph.
    """

    name: str = "kg_agent_create_node"

    kg_agent: object
    kg: GraphAdapter
    embeddings_adapter: EmbeddingsAdapter
    vector_store_adapter: VectorStoreAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "node": {
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
        "required": ["node"],
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
            "Use this tool to create a node in the knowledge graph. "
            "The argument should be a valid JSON object with a 'node' key. "
            "The 'node' key should be the node to create. "
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

        _node_arg = None
        if len(args) > 0:
            _node_arg = args[0]
        if len(kwargs) > 0:
            _node_arg = kwargs.get("node", {})
        if _node_arg is None:
            return "No node provided in the arguments or kwargs"
        try:
            _node = Node(**_node_arg)
        except Exception as e:
            return f"Error creating node: {e}"

        embedding_v = self.embeddings_adapter.embed_text(_node.name)
        vs = self.vector_store_adapter.add_vectors(
            [embedding_v], store="nodes", brain_id=self.brain_id
        )
        _node.properties = {
            **(_node.properties or {}),
            "v_id": vs[0],
        }

        self.kg.add_nodes([_node], brain_id=self.brain_id)

        return "Node created successfully"
