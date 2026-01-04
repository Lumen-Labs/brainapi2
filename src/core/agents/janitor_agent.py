"""
File: /janitor_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 2:28:14 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
from typing import List, Literal, Optional, Tuple
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

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.kg import Node
from src.constants.prompts.janitor_agent import (
    JANITOR_AGENT_NORMALIZE_INSERTION_PROMPT,
    JANITOR_AGENT_SYSTEM_PROMPT,
    JANITOR_AGENT_GRAPH_NORMALIZATOR_PROMPT,
    JANITOR_AGENT_GRAPH_NORMALIZATOR_SYSTEM_PROMPT,
)
from src.core.agents.architect_agent import (
    ArchitectAgentRelationship,
    ArchitectAgentNew,
)
from src.core.agents.scout_agent import ScoutEntity
from src.core.agents.tools.janitor_agent import (
    JanitorAgentGetSchemaTool,
    JanitorAgentSearchEntitiesTool,
    JanitorAgentExecuteGraphReadOperationTool,
    JanitorAgentExecuteGraphOperationTool,
)


class JanitorAgentInputOutput(BaseModel):
    """
    Janitor agent input and output, a single relationship or virtual node to normalize.
    Received directly from the Architect agent.
    """

    relationship: Optional[ArchitectAgentRelationship] = None
    virtual_node: Optional[ArchitectAgentNew] = None
    entity: Optional[ScoutEntity] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class JanitorAgent:
    """
    Janitor agent.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
        database_desc: str,
    ):
        self.llm_adapter = llm_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None
        self.database_desc = database_desc

    def _get_tools(self, brain_id: str = "default") -> List[BaseTool]:
        return [
            JanitorAgentGetSchemaTool(
                self,
                self.kg,
                brain_id=brain_id,
            ),
            JanitorAgentSearchEntitiesTool(
                self,
                self.kg,
                self.embeddings,
                self.vector_store,
                brain_id=brain_id,
            ),
            JanitorAgentExecuteGraphReadOperationTool(
                self,
                self.kg,
                self.database_desc,
                brain_id=brain_id,
            ),
        ]

    def _get_agent(
        self,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
        brain_id: str = "default",
        type_: str = Literal["janitor", "graph-janitor"],
    ):
        system_prompt = None
        if type_ == "janitor":
            system_prompt = JANITOR_AGENT_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        elif type_ == "graph-janitor":
            system_prompt = JANITOR_AGENT_GRAPH_NORMALIZATOR_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        else:
            raise ValueError(f"Invalid type: {type_}")

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(tools if tools else self._get_tools(brain_id)),
            system_prompt=system_prompt,
            response_format=output_schema if output_schema else None,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
        )

    def run_graph_normalizator(
        self,
        new_nodes: List[ScoutEntity | ArchitectAgentNew],
        text: str,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> Tuple[int, int]:
        """
        Runs a final janitor normalization across the kg snapshot
        """

        input_tokens = 0
        output_tokens = 0

        hops = self.kg.get_2nd_degree_hops(
            [n.uuid for n in new_nodes],
            flattened=True,
            vector_store_adapter=self.vector_store,
            brain_id=brain_id,
        )

        self._get_agent(
            tools=[
                JanitorAgentExecuteGraphOperationTool(
                    self,
                    self.kg,
                    self.database_desc,
                    brain_id=brain_id,
                )
            ],
            brain_id=brain_id,
            type_="graph-janitor",
        )

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": JANITOR_AGENT_GRAPH_NORMALIZATOR_PROMPT.format(
                                hops=hops,
                                text=text,
                            ),
                        }
                    ],
                },
                config={"recursion_limit": 25},
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
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Graph janitor normalizer agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Graph janitor normalizer agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        def _extract_tokens(r):
            response_metadata = r.get("response_metadata", {})
            token_usage = response_metadata.get("token_usage", {})
            if token_usage:
                return token_usage.get("prompt_tokens", 0), token_usage.get(
                    "completion_tokens", 0
                )
            usage_metadata = r.get("usage_metadata", {})
            if usage_metadata:
                return usage_metadata.get("input_tokens", 0), usage_metadata.get(
                    "output_tokens", 0
                )
            input_tokens = r.get("input_tokens", 0)
            output_tokens = r.get("output_tokens", 0)
            return input_tokens, output_tokens

        input_tokens, output_tokens = _extract_tokens(response)
        return input_tokens, output_tokens

    def run(
        self,
        input_output: JanitorAgentInputOutput,
        text: str,
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> JanitorAgentInputOutput:
        """
        Run the janitor agent.
        """

        self._get_agent(
            output_schema=JanitorAgentInputOutput, brain_id=brain_id, type_="janitor"
        )

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": JANITOR_AGENT_NORMALIZE_INSERTION_PROMPT.format(
                                unit_of_work=input_output.model_dump_json(),
                                text=text,
                                targeting=(
                                    f"""
                                The information is related to:
                                "{targeting.name}": {targeting.description}
                                {targeting.properties}
                                """
                                    if targeting
                                    else ""
                                ),
                            ),
                        }
                    ],
                },
                config={"recursion_limit": 25},
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
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Janitor agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        def _extract_tokens(r):
            response_metadata = r.get("response_metadata", {})
            token_usage = response_metadata.get("token_usage", {})
            if token_usage:
                return token_usage.get("prompt_tokens", 0), token_usage.get(
                    "completion_tokens", 0
                )
            usage_metadata = r.get("usage_metadata", {})
            if usage_metadata:
                return usage_metadata.get("input_tokens", 0), usage_metadata.get(
                    "output_tokens", 0
                )
            input_tokens = r.get("input_tokens", 0)
            output_tokens = r.get("output_tokens", 0)
            return input_tokens, output_tokens

        input_tokens, output_tokens = _extract_tokens(response)
        structured_response = response.get("structured_response")
        if structured_response:
            return JanitorAgentInputOutput(
                relationship=structured_response.relationship,
                virtual_node=structured_response.virtual_node,
                entity=structured_response.entity,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        else:
            return JanitorAgentInputOutput(
                relationship=None,
                virtual_node=None,
                entity=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
