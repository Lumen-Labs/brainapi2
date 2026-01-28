"""
File: /JanitorAgentSearchEntitiesTool.py
Created Date: Tuesday December 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 11:21:54 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from typing import Dict, List, Union
from langchain.tools import BaseTool

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import Node


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
        _queries = []
        if len(args) > 0:
            _queries = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            _queries = kwargs.get("queries", [])
            return "No queries provided in the arguments or kwargs"

        found_entities: Dict[str, Union[str, List[Node]]] = {}

        def _clean_query(query: str) -> str:
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
                    f.model_dump(mode="json") for f in founds
                ]

        return json.dumps(found_entities, indent=4)
