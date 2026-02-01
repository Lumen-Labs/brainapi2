"""
File: /ArchitectAgentCheckUsedEntitiesTool.py
Project: architect_agent
Created Date: Saturday January 17th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from langchain.tools import BaseTool

from src.utils.cleanup import strip_properties


class ArchitectAgentCheckUsedEntitiesTool(BaseTool):
    name: str = "architect_agent_check_used_entities"
    architect_agent: object

    def __init__(
        self,
        architect_agent: object,
    ):
        """
        Initialize the tool with the given architect agent and set its description.

        This tool inspects the architect agent's used_entities_set to report entities that have been used; it should be invoked after relationships are created and entities are marked as used.

        Parameters:
            architect_agent (object): The architect agent instance whose used entities the tool will check.
        """
        description: str = (
            "Tool for checking used entities. "
            "You must call this tool after calling the architect_agent_create_relationship tool and after marking entities as used."
            "Returns a list of entities that have been used."
        )
        super().__init__(
            architect_agent=architect_agent,
            description=description,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Collect the set of entities marked as used by the architect agent and return them as a JSON string.

        Each entity is converted to a plain mapping by calling `model_dump()` if present on the entity, otherwise `dict()`, and the resulting list of mappings is serialized to JSON.

        Returns:
            json_str (str): JSON-formatted string containing a list of serialized entity mappings.
        """
        entities_list = [
            (
                strip_properties([entity.model_dump(mode="json")])[0]
                if hasattr(entity, "model_dump")
                else entity
            )
            for entity in self.architect_agent.used_entities_dict.values()
        ]
        print("[DEBUG (architect_agent_check_used_entities)]: ", entities_list)
        return json.dumps(entities_list)
