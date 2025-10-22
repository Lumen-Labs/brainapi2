"""
File: /tool_schemas.py
Created Date: Wednesday October 22nd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday October 22nd 2025 10:04:19 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": (
                "The category of the node, "
                "identifying the type of node. "
                "(eg: Person, Organization, "
                "Location, Product, Service, Event, etc.)"
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "The name of the node, identifying the node. "
                "It is used to identify the node and is mandatory."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "The description of the node, providing additional context. "
                "Not mandatory but strongly recommended, "
                "optional only if you are unable to provide a "
                "description all the other cases is mandatory."
            ),
        },
        "properties": {
            "type": "object",
            "description": (
                "All the extra properties that will be added to the node "
                "and provide additional context."
            ),
            "additionalProperties": True,
        },
    },
    "required": ["label", "name"],
}

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": NODE_SCHEMA,
        "predicate": {
            "type": "string",
            "description": (
                "The description of the predicate of the triple. "
                "(can be a portion of the text where the predicate is found)"
            ),
        },
        "object": NODE_SCHEMA,
    },
    "required": ["subject", "predicate", "object"],
}
