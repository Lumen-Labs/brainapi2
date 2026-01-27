"""
File: /ArchitectAgentMarkEntitiesAsUsedTool.py
Project: architect_agent
Created Date: Saturday January 17th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 17th 2026 12:37:22 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from langchain.tools import BaseTool


class ArchitectAgentMarkEntitiesAsUsedTool(BaseTool):
    name: str = "architect_agent_mark_entities_as_used"
    architect_agent: object
    args_schema: dict = {
        "type": "object",
        "properties": {
            "entity_uuids": {
                "type": "array",
                "description": "A list of entity uuids to mark as used",
                "items": {
                    "type": "string",
                    "description": "The uuid of the entity to mark as used.",
                },
            },
        },
        "required": ["entity_uuids"],
    }

    def __init__(
        self,
        architect_agent: object,
    ):
        description: str = (
            "Tool that marks entities as used. "
            "Use this tool to mark entities as used. "
            "Input must be a list of entity uuids."
        )
        super().__init__(
            architect_agent=architect_agent,
            description=description,
        )

    def _run(self, *args, **kwargs) -> str:
        entities_to_mark = []

        if args and isinstance(args[0], dict) and "entity_uuids" in args[0]:
            entities_to_mark = args[0]["entity_uuids"]
        elif kwargs and "entity_uuids" in kwargs:
            entities_to_mark = kwargs["entity_uuids"]
        elif args and len(args) > 0 and isinstance(args[0], list):
            entities_to_mark = args[0]
        elif "entities" in kwargs:
            entities_to_mark = kwargs["entities"]
        elif "uuids" in kwargs:
            entities_to_mark = kwargs["uuids"]
        else:
            if len(kwargs) == 1:
                entities_to_mark = list(kwargs.values())[0]
            else:
                entities_to_mark = []

        if isinstance(entities_to_mark, str):
            entities_to_mark = [entities_to_mark]

        if entities_to_mark is None:
            entities_to_mark = []

        for entity_uuid in entities_to_mark:
            if entity_uuid in self.architect_agent.entities:
                removed_entity = self.architect_agent.entities.pop(entity_uuid)
                if removed_entity:
                    self.architect_agent.used_entities_set.append(removed_entity)
                else:
                    print(
                        "[DEBUG (architect_agent_mark_entities_as_used)]: Entity found but not removed: ",
                        entity_uuid,
                    )
            else:
                print(
                    "[DEBUG (architect_agent_mark_entities_as_used)]: Entity not found: ",
                    entity_uuid,
                )

        print(
            "[DEBUG (architect_agent_mark_entities_as_used)]: Used entities: ",
            self.architect_agent.used_entities_set,
            entities_to_mark,
        )

        return "OK"
