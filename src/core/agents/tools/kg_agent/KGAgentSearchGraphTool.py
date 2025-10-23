"""
File: /KGAgentSearchGraphTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:28:20 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node


class KGAgentSearchGraphTool(BaseTool):
    """
    Tool for searching the knowledge graph.
    """

    name: str = "kg_agent_search_graph"
    kg_agent: object
    kg: GraphAdapter
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None

    args_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Add this parameter if you want to search textually into the knowledge graph."
                ),
            },
            "nodes": {
                "type": "array",
                "description": (
                    "Add this parameter if you want to search for specific nodes "
                    "in the knowledge graph by their name and label."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "The label of the node to search for.",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the node to search for.",
                        },
                    },
                    "required": ["label", "name"],
                },
            },
        },
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        description: str = (
            "Tool for searching the knowledge graph. "
            "This tool will return the nodes with the properties "
            "and relationships that were found in the knowledge graph."
            "You can both search textually for semantically related nodes "
            "or search for specific nodes by their name and label."
            "The tool will return a list of nodes with the properties "
            "and 1st degree relationships that were found in the knowledge graph."
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=description,
            identification_params=identification_params or {},
            metadata=metadata or {},
        )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get("query", "")
        nodes = kwargs.get("nodes", [])
        _nodes = [Node(name=node["name"], label=node["label"]) for node in nodes]
        return self.kg.search_graph(nodes)
