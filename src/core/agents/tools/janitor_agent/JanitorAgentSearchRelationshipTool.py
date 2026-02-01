"""
File: /JanitorAgentSearchRelationshipTool.py
Project: janitor_agent
Created Date: Sunday February 1st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 1st 2026 2:40:43 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from typing import Dict, List, Tuple, Union
from langchain.tools import BaseTool

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.utils.cleanup import strip_properties


class JanitorAgentSearchRelationshipsTool(BaseTool):
    """
    Tool for searching the relationships of the knowledge graph.
    """

    name: str = "list_relationships"
    janitor_agent: object
    kg: GraphAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The uuid of the subject",
                        },
                        "object": {
                            "type": "string",
                            "description": "The uuid of the object",
                        },
                    },
                    "required": ["subject", "object"],
                },
                "description": "Lists the relationships between the subject and object.",
            },
        },
        "required": ["queries"],
    }

    def __init__(
        self,
        janitor_agent,
        kg: GraphAdapter,
        brain_id: str = "default",
    ):
        """
        Initialize the JanitorAgentSearchEntitiesTool with the required adapters and configuration.

        This tool performs plain-text searches against the knowledge graph and an optional vector store, aggregating results into a mapping from a sanitized query key to the list of matching entities.

        Parameters:
            brain_id (str): Identifier of the brain (knowledge scope) to use for searches. Defaults to "default".
        """
        description: str = (
            "Tool to list the relationships between the subject and object."
            "This tool will return the a list of relationships that were found in the knowledge graph between two entities."
            'Example input: { "queries": [{"subject": "123e4567-e89b-12d3-a456-426614174000", "object": "123e4567-e89b-12d3-a456-426614174000"}] }'
        )
        super().__init__(
            janitor_agent=janitor_agent,
            kg=kg,
            description=description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        List the relationships between the subject and object.

        Parameters:
            args: If provided, the first positional argument is treated as the list of relationships to process.
            kwargs:
                relationships (List[dict]): List of relationships to search for when not passed as a positional argument.

        Returns:
            str: A JSON string mapping each relationship to either a list of found relationship objects (each serialized to JSON) or a message indicating no matches were found.
        """

        _queries = []

        if len(args) > 0:
            arg_val = args[0]
            if isinstance(arg_val, list):
                _queries = arg_val
            elif isinstance(arg_val, str):
                try:
                    parsed = json.loads(arg_val)
                    if isinstance(parsed, dict) and parsed.get("queries"):
                        _queries = parsed.get("queries", [])
                    else:
                        _queries = [arg_val]
                except (json.JSONDecodeError, TypeError):
                    _queries = [arg_val]
            else:
                _queries = [arg_val]

        if not _queries and len(kwargs) > 0:
            _queries = kwargs.get("queries", [])

        if not isinstance(_queries, list):
            _queries = [_queries] if _queries else []

        if not _queries:
            return "No queries provided in the arguments or kwargs"

        found_relationships: Dict[
            str, Union[str, List[Tuple[Node, Predicate, Node]]]
        ] = {}

        for _query in _queries:
            founds = []
            kg_results = self.kg.list_relationships(
                brain_id=self.brain_id,
                subject=_query.get("subject"),
                object=_query.get("object"),
            )
            founds.extend(kg_results)

            if len(founds) == 0:
                found_relationships[
                    _query.get("subject") + " -> " + _query.get("object")
                ] = "Graph does not contain any relationships between the subject and object"
            else:
                found_relationships[
                    _query.get("subject") + " -> " + _query.get("object")
                ] = [
                    strip_properties(
                        [f.model_dump(mode="json")],
                        pop_also=["last_updated", "v_id", "properties"],
                    )[0]
                    for f in founds
                ]

        return json.dumps(found_relationships, indent=4)
