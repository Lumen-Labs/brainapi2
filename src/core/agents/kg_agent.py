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
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool

from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter


class KGAgent:
    """
    Knowledge Graph Agent. Used to update the knowledge graph with new information.
    Not used for batch updates.
    """

    def __init__(
        self, llm_adapter: LLMAdapter, cache_adapter: CacheAdapter, kg: GraphAdapter
    ):
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg

    async def _execute_graph_operation(self, operation: str) -> str:
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

    async def _get_tools(self) -> List[BaseTool]:
        return []

    async def _get_agent(self, prompt: str) -> str:
        _agent = create_react_agent(
            llm=self.llm_adapter.llm,
            prompt=prompt,
            tools=self._get_tools(),
        )
        agent = AgentExecutor(
            agent=_agent,
            llm=self.llm_adapter.llm,
            prompt=prompt,
            tools=self._get_tools(),
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=15,
            early_stopping_method="generate",
        )
        return agent

    async def search_kg(self, query: str) -> str:
        """
        Search the knowledge graph for information.
        """

    async def update_kg(self, information: str, metadata: Optional[dict]) -> str:
        """
        Update the knowledge graph with new information.
        """
