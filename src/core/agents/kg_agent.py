"""
File: /kg_agent.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:42:00 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
from langchain.agents import create_agent
from langchain.tools import BaseTool
from pydantic import BaseModel

from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.kg import Node
from src.constants.output_schemas import RetrieveNeighborsOutputSchema
from src.constants.prompts.kg_agent import (
    KG_AGENT_RETRIEVE_NEIGHBORS_PROMPT,
    KG_AGENT_SYSTEM_PROMPT,
    KG_AGENT_UPDATE_PROMPT,
    KG_AGENT_UPDATE_STRUCTURED_PROMPT,
)
from src.core.agents.tools.kg_agent import (
    KGAgentAddTripletsTool,
    KGAgentExecuteGraphOperationTool,
    KGAgentSearchGraphTool,
)
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter


class KGAgent:
    """
    Knowledge Graph Agent. Used to update the knowledge graph with new information.
    Not used for batch updates.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache_adapter: CacheAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
        database_desc: str,
    ):
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None
        self.database_desc = database_desc

    def _execute_graph_operation(self, operation: str) -> str:
        try:
            self.kg.execute_operation(operation)
            return f"Graph operation executed successfully: {operation}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error executing graph operation: {e}")
            return (
                f"Error executing graph operation: {e},"
                f"the db type is {type(self.kg.graphdb_type)}, "
                "use the appropriate syntax."
            )

    def _get_tools(self, identification_params: dict, metadata: dict) -> List[BaseTool]:
        return [
            KGAgentAddTripletsTool(
                self,
                self.kg,
                self.vector_store,
                self.embeddings,
                identification_params,
                metadata,
            ),
            KGAgentSearchGraphTool(
                self,
                self.kg,
                self.vector_store,
                self.embeddings,
                identification_params,
                metadata,
            ),
        ]

    def _get_agent(
        self,
        identification_params: dict,
        metadata: dict,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
    ):
        system_prompt = KG_AGENT_SYSTEM_PROMPT.format(
            extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
        )

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=tools if tools else self._get_tools(identification_params, metadata),
            system_prompt=system_prompt,
            response_format=output_schema if output_schema else None,
        )

    def search_kg(self, query: str) -> str:
        """
        Search the knowledge graph for information.
        """

    def update_kg(
        self,
        information: str,
        metadata: Optional[dict],
        identification_params: Optional[dict],
        preferred_entities: Optional[list[str]],
    ) -> str:
        """
        Update the knowledge graph with new information.
        """

        self._get_agent(identification_params, metadata)

        preferred_entities_prompt = f"""
        You must prioritize the extraction of the following entities: {preferred_entities}, 
        search and extract triplets including these entities first.
        """

        response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": KG_AGENT_UPDATE_PROMPT.format(
                            information=information,
                            preferred_entities=(
                                preferred_entities_prompt if preferred_entities else ""
                            ),
                            metadata=metadata,
                            identification_params=identification_params,
                        ),
                    }
                ],
            },
            print_mode="debug",
            config={"recursion_limit": 50},
        )
        return response

    def structured_update_kg(
        self, main_node: Node, textual_data: dict, identification_params: dict
    ) -> str:
        """
        Update the knowledge graph with new structured information.
        """

        self._get_agent(identification_params, metadata={})

        response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": KG_AGENT_UPDATE_STRUCTURED_PROMPT.format(
                            textual_data=textual_data,
                            main_node=main_node,
                        ),
                    }
                ],
            },
            print_mode="debug",
            config={"recursion_limit": 50},
        )
        return response

    def retrieve_neighbors(
        self, node: Node, looking_for: Optional[str], limit: int
    ) -> RetrieveNeighborsOutputSchema:
        """
        Retrieve the neighbors of a node.
        """

        graph_db_prop_keys = self.kg.get_graph_property_keys()
        graph_db_relationships = self.kg.get_graph_relationships()
        graph_db_entities = self.kg.get_graph_entities()

        extra_system_prompt_str = f"""
        The following are the information and schemas about the db, you must only use the following information to operate with the db:
        {{
            "property_keys": {graph_db_prop_keys},
            "relationships": {graph_db_relationships},
            "entities": {graph_db_entities},
        }}
        """

        self._get_agent(
            identification_params={},
            metadata={},
            tools=[
                KGAgentExecuteGraphOperationTool(
                    self,
                    self.kg,
                    self.database_desc,
                )
            ],
            output_schema=RetrieveNeighborsOutputSchema,
            extra_system_prompt=extra_system_prompt_str,
        )

        looking_for_prompt = f"""
        You must look for neighbors for the main node considering this reasons:
        {" ".join([f"- {reason}" for reason in looking_for])}
        """

        _response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": KG_AGENT_RETRIEVE_NEIGHBORS_PROMPT.format(
                            main_node=node, looking_for=looking_for_prompt, limit=limit
                        ),
                    }
                ],
            },
            config={"recursion_limit": 50},
            print_mode="debug",
        )

        response = _response.get("structured_response")

        return response
