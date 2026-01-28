"""
File: /janitor_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
from functools import reduce
from typing import List, Literal, Optional, Tuple, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import BaseTool
from langchain_core.messages import RemoveMessage
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
from src.constants.agents import AtomicJanitorAgentInputOutput, GraphConsolidatorOutput
from src.constants.kg import Node, Predicate
from src.constants.prompts.janitor_agent import (
    ATOMIC_JANITOR_AGENT_PROMPT,
    ATOMIC_JANITOR_AGENT_SYSTEM_PROMPT,
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
from src.utils.tokens import token_detail_from_token_counts


HISTORY_MAX_MESSAGES = 25
HISTORY_MAX_MESSAGES_DELETE = 8


class JanitorAgentInputOutput(BaseModel):
    """
    Janitor agent input and output, a single relationship or virtual node to normalize.
    Received directly from the Architect agent.
    """

    relationship: Optional[ArchitectAgentRelationship] = None
    virtual_node: Optional[ArchitectAgentNew] = None
    entity: Optional[ScoutEntity] = None


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
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.token_detail = None

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
        output_schemas: Optional[Union[BaseModel, Tuple[BaseModel, ...]]] = None,
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
                extra_system_prompt=extra_system_prompt if extra_system_prompt else "",
            )
        elif type_ == "atomic-janitor":
            system_prompt = ATOMIC_JANITOR_AGENT_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        else:
            raise ValueError(f"Invalid type: {type_}")

        response_format = None
        if output_schemas:
            if isinstance(output_schemas, tuple):
                if len(output_schemas) == 1:
                    union_schema = output_schemas[0]
                elif len(output_schemas) == 2:
                    union_schema = Union[output_schemas[0], output_schemas[1]]
                else:
                    union_schema = reduce(lambda x, y: Union[x, y], output_schemas)
            else:
                union_schema = output_schemas
            response_format = ToolStrategy(schema=union_schema)
        elif output_schema:
            if isinstance(output_schema, tuple):
                if len(output_schema) == 1:
                    union_schema = output_schema[0]
                elif len(output_schema) == 2:
                    union_schema = Union[output_schema[0], output_schema[1]]
                else:
                    union_schema = reduce(lambda x, y: Union[x, y], output_schema)
                response_format = ToolStrategy(schema=union_schema)
            else:
                response_format = output_schema

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(tools if tools else self._get_tools(brain_id)),
            system_prompt=system_prompt,
            response_format=response_format,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
        )

    def run_graph_consolidator(
        self,
        new_relationships: List[ArchitectAgentRelationship],
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> list[str]:
        """
        Runs a final janitor consolidation across the kg snapshot
        """

        accumulated_messages = []

        hops = self.kg.get_2nd_degree_hops(
            list(
                set(
                    [
                        *[r.tip.uuid for r in new_relationships],
                        *[r.tail.uuid for r in new_relationships],
                    ]
                )
            ),
            flattened=True,
            vector_store_adapter=self.vector_store,
            brain_id=brain_id,
            similarity_threshold=0.35,
        )

        self._get_agent(
            tools=[
                JanitorAgentExecuteGraphReadOperationTool(
                    self,
                    self.kg,
                    self.database_desc,
                    brain_id=brain_id,
                )
            ],
            output_schema=GraphConsolidatorOutput,
            brain_id=brain_id,
            type_="graph-janitor",
        )

        def _invoke_agent(previous_messages: list = None):
            from langchain_core.messages import HumanMessage

            messages_list = []
            if previous_messages:
                messages = previous_messages
                if len(messages) > HISTORY_MAX_MESSAGES:
                    delete_count = len(messages) - HISTORY_MAX_MESSAGES_DELETE
                    delete_ids = [msg.id for msg in messages[:delete_count]]
                    messages_list.extend(
                        [RemoveMessage(id=msg_id) for msg_id in delete_ids]
                    )
                    messages_list.extend(messages[delete_count:])
                else:
                    messages_list.extend(messages)

            user_message = HumanMessage(
                content=JANITOR_AGENT_GRAPH_NORMALIZATOR_PROMPT.format(
                    snapshot_json=json.dumps(hops),
                    units=json.dumps(
                        [rel.model_dump(mode="json") for rel in new_relationships]
                    ),
                )
            )
            messages_list.append(user_message)

            return self.agent.invoke(
                {"messages": messages_list},
                config={"recursion_limit": 100},
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent, previous_messages)
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
                    f"Graph janitor normalizer agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(previous_messages: list):
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(messages)
                return response
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Graph janitor normalizer agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        try:
            response = _invoke_and_process(accumulated_messages)
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Graph janitor normalizer agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        structured_response = response.get("structured_response")
        if structured_response:
            return structured_response.tasks
        else:
            return []

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

        accumulated_messages = []

        def _invoke_agent(previous_messages: list = None):
            from langchain_core.messages import HumanMessage

            messages_list = []
            if previous_messages:
                messages = previous_messages
                if len(messages) > HISTORY_MAX_MESSAGES:
                    delete_count = len(messages) - HISTORY_MAX_MESSAGES_DELETE
                    delete_ids = [msg.id for msg in messages[:delete_count]]
                    messages_list.extend(
                        [RemoveMessage(id=msg_id) for msg_id in delete_ids]
                    )
                    messages_list.extend(messages[delete_count:])
                else:
                    messages_list.extend(messages)

            user_message = HumanMessage(
                content=JANITOR_AGENT_NORMALIZE_INSERTION_PROMPT.format(
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
                )
            )
            messages_list.append(user_message)

            return self.agent.invoke(
                {"messages": messages_list},
                config={"recursion_limit": 100},
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent, previous_messages)
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
                    f"Janitor agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(previous_messages: list):
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(messages)
                return response
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        try:
            response = _invoke_and_process(accumulated_messages)
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise
        structured_response = response.get("structured_response")
        if structured_response:
            return JanitorAgentInputOutput(
                relationship=structured_response.relationship,
                virtual_node=structured_response.virtual_node,
                entity=structured_response.entity,
            )
        else:
            return JanitorAgentInputOutput(
                relationship=None,
                virtual_node=None,
                entity=None,
            )

    def run_atomic_janitor(
        self,
        input_relationships: List[ArchitectAgentRelationship],
        text: str,
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> AtomicJanitorAgentInputOutput | str:
        """
        Run the atomic janitor agent.
        Can return istructions on how to correct the wrong relationships or 'OK' if the relationships are correct.
        """

        self._get_agent(
            output_schemas=AtomicJanitorAgentInputOutput,
            brain_id=brain_id,
            type_="atomic-janitor",
        )

        accumulated_messages = []

        def _invoke_agent(previous_messages: list = None):
            from langchain_core.messages import HumanMessage

            messages_list = []
            if previous_messages:
                messages = previous_messages
                if len(messages) > HISTORY_MAX_MESSAGES:
                    delete_count = len(messages) - HISTORY_MAX_MESSAGES_DELETE
                    delete_ids = [msg.id for msg in messages[:delete_count]]
                    messages_list.extend(
                        [RemoveMessage(id=msg_id) for msg_id in delete_ids]
                    )
                    messages_list.extend(messages[delete_count:])
                else:
                    messages_list.extend(messages)

            user_message = HumanMessage(
                content=ATOMIC_JANITOR_AGENT_PROMPT.format(
                    units_of_work=json.dumps(
                        [rel.model_dump(mode="json") for rel in input_relationships]
                    ),
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
                )
            )
            messages_list.append(user_message)

            return self.agent.invoke(
                {"messages": messages_list},
                config={"recursion_limit": 100},
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent, previous_messages)
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
                    f"Atomic janitor agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Atomic janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except:
                raise

        def _invoke_and_process(previous_messages: list):
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(messages)
                return response
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Atomic janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        try:
            response = _invoke_and_process(accumulated_messages)
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Atomic janitor agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        structured_response = response.get("structured_response")

        if structured_response and structured_response.status == "OK":
            return "OK"
        else:
            return structured_response

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
