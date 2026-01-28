"""
File: /ArchitectAgentGetRemainingEntitiesToProcessTool.py
Project: architect_agent
Created Date: Friday January 16th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday January 16th 2026 10:55:16 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from langchain.tools import BaseTool


class ArchitectAgentGetRemainingEntitiesToProcessTool(BaseTool):
    name: str = "architect_agent_get_remaining_entities_to_process"
    architect_agent: object

    def __init__(
        self,
        architect_agent: object,
    ):
        """
        Initialize the tool with an architect agent whose entities will be inspected to determine remaining work.
        
        Parameters:
            architect_agent (object): The architect agent instance that exposes an `entities` mapping; each entity is expected to support `model_dump(mode="json")` for serialization.
        """
        description: str = (
            "Tool for getting the remaining entities to process. "
            "Use this tool to get the remaining entities to process. "
            "Returns a list of entities that are not yet processed"
            "or the initial list of entities for the first call."
        )
        super().__init__(
            architect_agent=architect_agent,
            description=description,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Get a JSON string listing the remaining entities to process.
        
        Returns:
            str: A JSON array string where each element is the serialized representation of an entity produced by calling `model_dump(mode="json")` on each entity in `self.architect_agent.entities`.
        """
        remaining_entities = json.dumps(
            [
                entity.model_dump(mode="json")
                for entity in self.architect_agent.entities.values()
            ]
        )
        print(
            "[DEBUG (architect_agent_get_remaining_entities_to_process)]: ",
            remaining_entities,
        )
        return remaining_entities