"""
File: /get_schema.py
Created Date: Tuesday December 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 11:07:15 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter


class JanitorAgentGetSchemaTool(BaseTool):
    """
    Tool for getting the schema of the knowledge graph.
    """

    name: str = "get_schema"
    janitor_agent: object
    kg: GraphAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "The target to get the schema of. (node_labels or relationship_types)",
                "enum": ["node_labels", "relationship_types", "event_names"],
            },
        },
        "required": ["target"],
    }

    def __init__(
        self,
        janitor_agent,
        kg: GraphAdapter,
        brain_id: str = "default",
    ):
        description: str = (
            "Tool for getting the schema of the knowledge graph. "
            "This tool will return the schema/ontology of the knowledge graph."
        )
        super().__init__(
            janitor_agent=janitor_agent,
            kg=kg,
            description=description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        _target = kwargs.get("target", "node_labels").lower()

        schema_result = self.kg.get_schema(brain_id=self.brain_id)

        if _target == "node_labels":
            labels = schema_result.get("labels", [])
            return json.dumps(labels, indent=4)
        elif _target == "relationship_types":
            relationships = schema_result.get("relationships", [])
            return json.dumps(relationships, indent=4)
        elif _target == "event_names":
            event_names = schema_result.get("event_names", [])
            return json.dumps(event_names, indent=4)
        return "Invalid target parameter. It must be either 'node_labels' or 'relationship_types' or 'event_names'"
