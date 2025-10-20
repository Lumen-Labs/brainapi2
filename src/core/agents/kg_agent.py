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

from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.prompts.kg_agent import (
    KG_AGENT_SYSTEM_PROMPT,
    KG_AGENT_UPDATE_PROMPT,
)
from src.core.agents.tools.kg_agent import KGAgentExecuteGraphOperationTool


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
        self.agent = None

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

    def _get_tools(self) -> List[BaseTool]:
        return [
            KGAgentExecuteGraphOperationTool(self, self.kg, self.kg.graphdb_description)
        ]

    def _get_agent(self):
        if self.agent is None:
            system_prompt = KG_AGENT_SYSTEM_PROMPT

            self.agent = create_agent(
                model=self.llm_adapter.llm.langchain_model,
                tools=self._get_tools(),
                system_prompt=system_prompt,
            )
        return self.agent

    def search_kg(self, query: str) -> str:
        """
        Search the knowledge graph for information.
        """
        # TODO vector search and retrieval

    def update_kg(self, information: str, metadata: Optional[dict]) -> str:
        """
        Update the knowledge graph with new information.
        """

        self._get_agent()
        response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": KG_AGENT_UPDATE_PROMPT.format(
                            information=information,
                            metadata=metadata,
                        ),
                    }
                ],
            }
        )
        return response
