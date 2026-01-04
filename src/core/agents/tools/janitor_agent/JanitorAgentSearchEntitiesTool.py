"""
File: /JanitorAgentSearchEntitiesTool.py
Created Date: Tuesday December 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 11:21:54 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
from typing import List
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
            "query": {
                "type": "string",
                "description": "The plain text query to search for entities in the knowledge graph within their names.",
            },
        },
        "required": ["query"],
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
            "Tool for searching the entities of the knowledge graph by plain text name. "
            "This tool will return the entities that were found in the knowledge graph."
            "Example input: { 'query': 'John Doe' }"
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
        _query = ""
        if len(args) > 0:
            _query = args[0]
        else:
            _query = kwargs.get("query", "")

        if len(_query) == 0:
            return "No query provided in the arguments or kwargs"

        kg_results = self.kg.search_entities(brain_id=self.brain_id, query_text=_query)

        found_entities: List[Node] = []

        for kg_result in kg_results.results:
            found_entities.append(kg_result)

        try:
            query_vector = self.embeddings.embed_text(_query)
            if query_vector.embeddings and len(query_vector.embeddings) > 0:
                v_results = self.vector_store.search_vectors(
                    query_vector.embeddings,
                    store="nodes",
                    brain_id=self.brain_id,
                )
                if len(v_results) > 0:
                    found_entities.extend(
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

        if len(found_entities) == 0:
            return "Knowledge graph does not contain any entities that match the query"

        return json.dumps(
            [entity.model_dump(mode="json") for entity in found_entities], indent=4
        )
