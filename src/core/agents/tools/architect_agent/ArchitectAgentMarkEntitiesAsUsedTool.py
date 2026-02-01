"""
File: /ArchitectAgentMarkEntitiesAsUsedTool.py
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
        """
        Initialize the tool with a reference to the architect agent whose entities may be marked as used.

        Parameters:
            architect_agent (object): The architect agent instance whose `entities` and `used_entities_set` this tool will update.
        """
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
        """
        Mark the given entity UUIDs as used on the associated architect agent.

        Accepts several input shapes to identify the UUIDs to mark: a dict argument with the key "entity_uuids", a keyword "entity_uuids", a list passed as the first positional argument, or keywords "entities" or "uuids". If a single string is provided it will be treated as a single UUID. If no recognizable input is provided, no entities are marked.

        Parameters:
            *args: Positional arguments; supported forms include:
                - A dict containing the key "entity_uuids" with a list of UUID strings.
                - A list of UUID strings as the first positional argument.
            **kwargs: Keyword arguments; supported keys include:
                - "entity_uuids", "entities", or "uuids" with a list (or single string) of UUIDs.
                - If exactly one keyword is provided, its value is used as the UUID list.

        Returns:
            result (str): The string "OK" after processing.

        Side effects:
            For each provided UUID found in self.architect_agent.entities, the entity is removed from that mapping and appended to self.architect_agent.used_entities_set.
        """
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
                    _ent = (
                        removed_entity.model_dump(mode="json")
                        if hasattr(removed_entity, "model_dump")
                        else removed_entity
                    )
                    self.architect_agent.used_entities_dict[_ent["uuid"]] = (
                        strip_properties([_ent])[0]
                    )
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
            list(self.architect_agent.used_entities_dict.keys()),
            entities_to_mark,
        )

        return "OK"
