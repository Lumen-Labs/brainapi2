"""
File: /kg_agent.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Iterable, List, Literal, Optional, Union
from langchain.agents import create_agent
from langchain.tools import BaseTool
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.kg import Node
from src.constants.output_schemas import RetrieveNeighborsOutputSchema
from src.constants.prompts.kg_agent import (
    KG_AGENT_GRAPH_CONSOLIDATOR_PROMPT,
    KG_AGENT_GRAPH_CONSOLIDATOR_SYSTEM_PROMPT,
    KG_AGENT_RETRIEVE_NEIGHBORS_PROMPT,
    KG_AGENT_SYSTEM_PROMPT,
    KG_AGENT_UPDATE_PROMPT,
    KG_AGENT_UPDATE_STRUCTURED_PROMPT,
)
from src.core.agents.tools.kg_agent import (
    KGAgentAddTripletsTool,
    KGAgentDeleteRelationshipTool,
    KGAgentExecuteGraphOperationTool,
    KGAgentSearchGraphTool,
    KGAgentUpdatePropertiesTool,
)
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.utils.tokens import token_detail_from_token_counts


class KGAgent:
    """
    Knowledge Graph Agent. Used to operate with the knowledge graph with new information.
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
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.token_detail = None

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

    def _get_tools(
        self, identification_params: dict, metadata: dict, brain_id: str = "default"
    ) -> List[BaseTool]:
        return [
            KGAgentAddTripletsTool(
                self,
                self.kg,
                self.vector_store,
                self.embeddings,
                identification_params,
                metadata,
                brain_id=brain_id,
            ),
            KGAgentSearchGraphTool(
                self,
                self.kg,
                self.vector_store,
                self.embeddings,
                identification_params,
                metadata,
                brain_id=brain_id,
            ),
            KGAgentDeleteRelationshipTool(
                self,
                self.kg,
                self.vector_store,
                brain_id=brain_id,
            ),
            KGAgentUpdatePropertiesTool(
                self,
                self.kg,
                brain_id=brain_id,
            ),
        ]

    def _get_agent(
        self,
        type_: Literal["normal", "graph-consolidator"],
        identification_params: dict,
        metadata: dict,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
        brain_id: str = "default",
    ):
        system_prompt = None
        if type_ == "normal":
            system_prompt = KG_AGENT_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        elif type_ == "graph-consolidator":
            system_prompt = KG_AGENT_GRAPH_CONSOLIDATOR_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(
                tools
                if tools
                else self._get_tools(identification_params, metadata, brain_id)
            ),
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
        brain_id: str = "default",
    ) -> str:
        """
        Update the knowledge graph with new information.
        """

        self._get_agent(identification_params, metadata, brain_id=brain_id)

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
        self,
        main_node: Node,
        textual_data: dict,
        identification_params: dict,
        brain_id: str = "default",
    ) -> str:
        """
        Update the knowledge graph with new structured information.
        """

        self._get_agent(identification_params, metadata={}, brain_id=brain_id)

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
        self, node: Node, looking_for: Optional[Union[str, Iterable[str]]], limit: int
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

        if isinstance(looking_for, str):
            looking_for_items = [looking_for]
        else:
            looking_for_items = list(looking_for) if looking_for else []

        looking_for_prompt = f"""
        You must look for neighbors for the main node considering this reasons:
        {" ".join([f"- {reason}" for reason in looking_for_items])}
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

    def run_graph_consolidator_operator(
        self,
        task: str,
        brain_id: str = "default",
        timeout: int = 180,
        max_retries: int = 3,
    ) -> str:
        """
        Run the graph consolidator operator.
        """

        self._get_agent(
            type_="graph-consolidator",
            identification_params={},
            metadata={},
            tools=[
                KGAgentExecuteGraphOperationTool(
                    self,
                    self.kg,
                    self.database_desc,
                    brain_id=brain_id,
                )
            ],
        )

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": KG_AGENT_GRAPH_CONSOLIDATOR_PROMPT.format(
                                task=task
                            ),
                        }
                    ],
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry():
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent)
                    response = future.result(timeout=timeout)
                    for m in response.get("messages", []):
                        if hasattr(m, "usage_metadata"):
                            self._update_token_counts(m.usage_metadata)
                            self.token_detail = token_detail_from_token_counts(
                                self.input_tokens,
                                self.output_tokens,
                                self.cached_tokens,
                                self.reasoning_tokens,
                            )
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Graph consolidator operator invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Graph consolidator operator invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        return "OK"

    def _update_token_counts(self, usage_metadata: dict):
        """Extract all token counts from usage metadata"""
        # Base counts
        self.input_tokens += usage_metadata.get("input_tokens", 0)
        self.output_tokens += usage_metadata.get("output_tokens", 0)

        # Input details (caching)
        input_details = usage_metadata.get("input_token_details", {})
        self.cached_tokens += input_details.get("cache_read", 0)

        # Output details (reasoning)
        output_details = usage_metadata.get("output_token_details", {})
        self.reasoning_tokens += output_details.get("reasoning", 0)
