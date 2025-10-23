"""
File: /KGAgentAddNodesTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:27:01 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import uuid
from typing import Optional
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node
from src.services.api.constants.tool_schemas import NODE_SCHEMA


class KGAgentAddNodesTool(BaseTool):
    """
    Tool for adding nodes to the knowledge graph.
    """

    name: str = "kg_agent_add_nodes"

    kg_agent: object
    kg: GraphAdapter
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None

    args_schema: dict = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": NODE_SCHEMA,
            },
        },
        "required": ["nodes"],
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        description: str = (
            "Tool specifically for adding nodes to the knowledge graph. "
            "This tool will return the nodes that were added to the knowledge graph."
            "Input should be a valid JSON object with a 'nodes' key. "
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=description,
            identification_params=identification_params or {},
            metadata=metadata or {},
        )

    def _run(self, *args, **kwargs) -> str:
        nodes = [
            Node(**node, uuid=str(uuid.uuid4())) for node in kwargs.get("nodes", [])
        ]
        self.kg.add_nodes(nodes, self.identification_params, self.metadata)
        return "Nodes added successfully"
