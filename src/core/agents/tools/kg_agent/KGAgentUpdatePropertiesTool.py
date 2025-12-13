"""
File: /KGAgentUpdatePropertiesTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:29:08 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter


class KGAgentUpdatePropertiesTool(BaseTool):
    """
    Tool for updating the properties of a node or relationship in the knowledge graph.
    """

    name: str = "kg_agent_update_properties"
    kg_agent: object
    kg: GraphAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "updating": {
                "type": "string",
                "description": ("What to update, a relationship or a node"),
                "enum": ["relationship", "node"],
            },
            "uuid": {
                "type": "string",
                "description": ("The UUID of the node or relationship to update."),
            },
            "new_properties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "property": {
                            "type": "string",
                            "description": ("The property to update."),
                        },
                        "value": {
                            "type": "string",
                            "description": ("The new value of the property."),
                        },
                    },
                },
                "description": ("The new properties to update."),
            },
            "properties_to_remove": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": ("The properties to remove."),
                },
                "description": ("The properties to remove."),
            },
        },
        "required": ["updating", "uuid", "properties"],
    }

    def __init__(
        self,
        kg_agent: object,
        kg: GraphAdapter,
        brain_id: str = "default",
    ):
        description: str = (
            "This tool is used to update the properties of a node or relationship in the knowledge graph. "
            "Returns the node or relationship that was updated in the knowledge graph if it was found else returns an error message."
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=description,
            brain_id=brain_id or "default",
        )

    def _run(self, *args, **kwargs) -> str:
        updating = kwargs.get("updating", "node")
        uuid = kwargs.get("uuid", "")
        new_properties = kwargs.get("new_properties", {})
        properties_to_remove = kwargs.get("properties_to_remove", [])

        updated_node_or_relationship = self.kg.update_properties(
            uuid=uuid,
            updating=updating,
            brain_id=self.brain_id,
            new_properties=new_properties,
            properties_to_remove=properties_to_remove,
        )

        if updated_node_or_relationship:
            return f"Node or relationship updated successfully: {updated_node_or_relationship}"
        else:
            return "Node or relationship not found"
