"""
File: /JanitorAgentSearchEntitiesTool.py
Created Date: Tuesday December 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from typing import Dict, List, Union
from langchain.tools import BaseTool

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node
from src.utils.cleanup import strip_properties


class JanitorAgentSearchEntitiesTool(BaseTool):
    """
    Tool for searching the entities of the knowledge graph.
    """

    name: str = "search_entities"
    janitor_agent: object
    kg: GraphAdapter
    embeddings: EmbeddingsAdapter
    vector_store: VectorStoreAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The plain text query to search for entities in the knowledge graph within their names.",
                },
                "description": "The list of things to search for in the knowledge graph.",
            },
        },
        "required": ["queries"],
    }

    def __init__(
        self,
        janitor_agent,
        kg: GraphAdapter,
        embeddings: EmbeddingsAdapter,
        vector_store: VectorStoreAdapter,
        brain_id: str = "default",
    ):
        """
        Initialize the JanitorAgentSearchEntitiesTool with the required adapters and configuration.

        This tool performs plain-text searches against the knowledge graph and an optional vector store, aggregating results into a mapping from a sanitized query key to the list of matching entities.

        Parameters:
            brain_id (str): Identifier of the brain (knowledge scope) to use for searches. Defaults to "default".
        """
        description: str = (
            "Tool to search for entities in the knowledge graph by plain text names. "
            "Call this tool once to search for a list of entities with different names, "
            "don't recall the tool with different queries for same entities, be exaustive on the first call."
            "This tool will return the a map of entities that were found in the knowledge graph where the key is the query and the value is the list of entities that were found."
            'Example input: { "queries": ["John Doe", "John", "John D.", "Mary", "John\'s Wife"] }'
        )
        super().__init__(
            janitor_agent=janitor_agent,
            kg=kg,
            embeddings=embeddings,
            vector_store=vector_store,
            description=description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Search multiple query strings for matching entities in the knowledge graph and vector store, and return the aggregated results as a JSON-formatted mapping keyed by a sanitized query string.

        Parameters:
            args: If provided, the first positional argument is treated as the list of query strings to process.
            kwargs:
                queries (List[str]): List of query strings to search for when not passed as a positional argument.

        Returns:
            str: A JSON string mapping each sanitized query to either a list of found entity objects (each serialized to JSON) or a message indicating no matches were found.
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

        found_entities: Dict[str, Union[str, List[Node]]] = {}

        def _clean_query(query: str) -> str:
            """
            Sanitizes a query string into a normalized key suitable for dictionary lookup.

            Removes common punctuation, trims leading/trailing whitespace, converts to lowercase,
            and replaces spaces and hyphens with underscores.

            Parameters:
                query (str): The input query text to normalize.

            Returns:
                sanitized (str): The normalized query string with punctuation removed, spaces and hyphens replaced by underscores, trimmed and lowercased.
            """
            chars_to_remove = "'.\"()[]{}\\|*/+=<>,!?:;"
            chars_to_underscore = " -"
            translation_table = str.maketrans(
                chars_to_underscore, "_" * len(chars_to_underscore), chars_to_remove
            )
            return query.strip().lower().translate(translation_table).replace(" ", "_")

        for _query in _queries:
            founds = []
            kg_results = self.kg.search_entities(
                brain_id=self.brain_id, query_text=_query
            )
            founds.extend(kg_results.results)
            try:
                query_vector = self.embeddings.embed_text(_query)
                if query_vector.embeddings and len(query_vector.embeddings) > 0:
                    v_results = self.vector_store.search_vectors(
                        query_vector.embeddings,
                        store="nodes",
                        brain_id=self.brain_id,
                        k=3,
                    )
                    if len(v_results) > 0:
                        founds.extend(
                            self.kg.get_nodes_by_uuid(
                                [
                                    v_result.metadata.get("uuid")
                                    for v_result in v_results
                                    if v_result.metadata.get("uuid") is not None
                                ],
                                brain_id=self.brain_id,
                            )
                        )
            except Exception as e:
                print(f"Vector search failed, using KG results only: {e}")

            if len(founds) == 0:
                found_entities[_clean_query(_query)] = (
                    "Knowledge graph does not contain any entities that match the query"
                )
            else:
                found_entities[_clean_query(_query)] = [
                    strip_properties(
                        [f.model_dump(mode="json")],
                        pop_also=["last_updated", "v_id", "properties"],
                    )[0]
                    for f in founds
                ]

        return json.dumps(found_entities, indent=4)
