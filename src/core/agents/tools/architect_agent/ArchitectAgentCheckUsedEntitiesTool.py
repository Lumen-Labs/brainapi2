"""
File: /ArchitectAgentCheckUsedEntitiesTool.py
Project: architect_agent
Created Date: Saturday January 17th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 17th 2026 12:37:32 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from langchain.tools import BaseTool


class ArchitectAgentCheckUsedEntitiesTool(BaseTool):
    name: str = "architect_agent_check_used_entities"
    architect_agent: object

    def __init__(
        self,
        architect_agent: object,
    ):
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
        entities_list = [
            entity.model_dump() if hasattr(entity, "model_dump") else entity.dict()
            for entity in self.architect_agent.used_entities_set
        ]
        print("[DEBUG (architect_agent_check_used_entities)]: ", entities_list)
        return json.dumps(entities_list)
