"""
File: /janitor_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
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
from src.core.agents.tools.janitor_agent.JanitorAgentSearchRelationshipTool import (
    JanitorAgentSearchRelationshipsTool,
)
from src.utils.cleanup import strip_properties
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
        """
        Initialize the JanitorAgent with required adapters and a database description.

        Parameters:
                llm_adapter (LLMAdapter): Adapter for language model interactions.
                kg (GraphAdapter): Adapter for graph database operations.
                vector_store (VectorStoreAdapter): Adapter for vector search and retrieval.
                embeddings (EmbeddingsAdapter): Adapter to compute embeddings for text.
                database_desc (str): Human-readable description of the target database.

        Initializes internal token counters and token detail storage.
        """
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
        """
        Assembles the set of tools the Janitor agent uses, scoped to the specified brain.

        Parameters:
            brain_id (str): Identifier of the brain/context used to instantiate each tool.

        Returns:
            List[BaseTool]: A list containing the schema access tool, the entity search tool (using embeddings and vector store), and the graph read-operation tool configured for this agent and brain_id.
        """
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
            JanitorAgentSearchRelationshipsTool(
                self,
                self.kg,
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
        """
        Configure and instantiate the internal LangChain agent and assign it to `self.agent`.

        Constructs a system prompt based on `type_` ("janitor", "graph-janitor", or "atomic-janitor"), derives a response schema from `output_schema` or `output_schemas` (merging multiple schemas into a Union when needed and wrapping complex schemas in `ToolStrategy`), and calls `create_agent` with the selected model, tools, system prompt, response format, and debug flag.

        Parameters:
            tools (Optional[List[BaseTool]]): Optional list of tools to expose to the agent; if omitted, the agent's default tools for `brain_id` are used.
            output_schema (Optional[BaseModel]): Single response schema or a tuple of schemas to use as the agent's response format.
            output_schemas (Optional[Union[BaseModel, Tuple[BaseModel, ...]]]): Alternative parameter allowing multiple output schemas; when provided, schemas are unified into a Union and wrapped in `ToolStrategy`.
            extra_system_prompt (Optional[dict]): Optional dictionary whose contents are formatted into the selected system prompt.
            brain_id (str): Identifier for the knowledge brain used when resolving default tools.
            type_ (str): Agent mode controlling system prompt selection; supported values are "janitor", "graph-janitor", and "atomic-janitor".

        """
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

    def _content_only_history(
        self, messages: Optional[list], keep_last: Optional[int] = None
    ) -> list:
        if not messages:
            return []
        pruned = messages
        if keep_last is not None and len(pruned) > keep_last:
            pruned = pruned[-keep_last:]
        content_only = []
        for msg in pruned:
            if isinstance(msg, dict):
                content = msg.get("content")
                role = msg.get("role") or "assistant"
            else:
                content = getattr(msg, "content", None)
                msg_type = getattr(msg, "type", None)
                if msg_type in ("human", "user"):
                    role = "user"
                elif msg_type == "system":
                    role = "system"
                else:
                    role = "assistant"
            if content is None:
                continue
            content_only.append({"role": role, "content": content})
        return content_only

    def run_graph_consolidator(
        self,
        new_relationships: List[ArchitectAgentRelationship],
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> list[str]:
        """
        Run graph-level consolidation for the provided relationships against a KG snapshot.

        Parameters:
            new_relationships (List[ArchitectAgentRelationship]): Relationships to normalize and consolidate into the graph.
            brain_id (str): Identifier of the knowledge brain/context to use.
            timeout (int): Maximum seconds to wait for a single agent invocation before timing out.
            max_retries (int): Number of retry attempts for agent invocation on timeout.

        Returns:
            list[str]: Consolidation tasks produced by the graph janitor; returns an empty list if no tasks were produced.

        Raises:
            TimeoutError: If the agent invocation times out or fails after the configured number of retries.
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
            """
            Builds a message history (pruning older entries when it exceeds HISTORY_MAX_MESSAGES), appends a user message containing a graph-normalization prompt populated with the current hops snapshot and the provided new relationships, and invokes the configured agent with that message sequence.

            Parameters:
                previous_messages (list): Optional sequence of prior messages to include in the history. When the sequence length exceeds HISTORY_MAX_MESSAGES, older messages are replaced with RemoveMessage objects to trim the history.

            Returns:
                The agent's response object returned by self.agent.invoke for the assembled messages.
            """
            messages_list = self._content_only_history(
                previous_messages, keep_last=HISTORY_MAX_MESSAGES_DELETE
            )
            messages_list.append(
                {
                    "role": "user",
                    "content": JANITOR_AGENT_GRAPH_NORMALIZATOR_PROMPT.format(
                        snapshot_json=json.dumps(hops),
                        units=json.dumps(
                            strip_properties(
                                [
                                    rel.model_dump(mode="json")
                                    for rel in new_relationships
                                ]
                            )
                        ),
                    ),
                }
            )

            return self.agent.invoke(
                {"messages": messages_list},
                config={
                    "recursion_limit": 100,
                    "tags": ["janitor_agent"],
                    "metadata": {"agent": "janitor_agent", "brain_id": brain_id},
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            """
            Invoke the configured agent in a thread pool, update the agent's token counters from any returned message usage metadata, and return the agent response.

            Parameters:
                previous_messages (list): The accumulated message history to provide to the agent for this invocation.

            Returns:
                dict: The agent's response object (expected to contain a "messages" list and optional usage metadata).

            Raises:
                TimeoutError: If the agent invocation does not complete within the configured timeout.
            """
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
                                "janitor_agent",
                            )
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Graph janitor normalizer agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(previous_messages: list):
            """
            Invoke the agent with retry handling and append any returned messages to the accumulated_messages history.

            Parameters:
                previous_messages (list): The list of messages to send to the agent as the current context.

            Returns:
                dict: The raw response object returned by the agent invocation.

            Raises:
                TimeoutError: If the agent invocation exhausts retries or a timeout occurs; the exception message includes the last attempt's error.
            """
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(self._content_only_history(messages))
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
        Normalize a single insertion unit using the janitor agent.

        Parameters:
            input_output (JanitorAgentInputOutput): The unit of work to normalize (relationship, virtual node, or entity).
            text (str): Human-provided text that guides the normalization.
            targeting (Optional[Node]): Optional human-readable context for a Node target (includes name, description, and properties).
            brain_id (str): Identifier of the brain/knowledge scope to use.
            timeout (int): Maximum seconds to wait for a single agent invocation before raising a TimeoutError.
            max_retries (int): Maximum number of retry attempts for transient invocation timeouts.

        Returns:
            JanitorAgentInputOutput: A container populated with `relationship`, `virtual_node`, and/or `entity` extracted from the agent's structured response; fields are set to `None` when not provided by the agent.

        Raises:
            TimeoutError: If the agent invocation times out or fails after the configured number of retries.
        """

        self._get_agent(
            output_schema=JanitorAgentInputOutput, brain_id=brain_id, type_="janitor"
        )

        accumulated_messages = []

        def _invoke_agent(previous_messages: list = None):
            """
            Builds a message list (with optional history pruning) and invokes the configured agent with a normalization prompt.

            Parameters:
                previous_messages (list | None): Prior conversation messages to include in the invocation; when provided and longer than HISTORY_MAX_MESSAGES, older messages are removed via deletion markers before sending.

            Returns:
                agent_response: The result returned by the agent's invoke call (the agent's response object).
            """
            messages_list = self._content_only_history(
                previous_messages, keep_last=HISTORY_MAX_MESSAGES_DELETE
            )
            messages_list.append(
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
            )

            return self.agent.invoke(
                {"messages": messages_list},
                config={
                    "recursion_limit": 100,
                    "tags": ["janitor_agent"],
                    "metadata": {"agent": "janitor_agent", "brain_id": brain_id},
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            """
            Invoke the configured agent with the provided message history, update the agent's token counters from any returned message usage metadata, and return the raw agent response.

            Parameters:
                previous_messages (list): Accumulated conversation messages to send to the agent.

            Returns:
                dict: The agent's raw response object (may contain a "messages" list and other metadata).

            Raises:
                TimeoutError: If the agent call exceeds the configured timeout.
            """
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
                                "janitor_agent",
                            )
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Janitor agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(previous_messages: list):
            """
            Invoke the agent with retry logic and process its response into the accumulated message history.

            Parameters:
                previous_messages (list): The message history passed to the agent for this invocation.

            Returns:
                dict: The raw agent response object returned by the retry wrapper; may contain a "messages" key whose items are appended to the outer `accumulated_messages`.

            Raises:
                TimeoutError: If the retry wrapper exhausts attempts or a timeout occurs; when retries are exhausted the exception message includes the last attempt number and its error.
            """
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(self._content_only_history(messages))
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
        Validate and produce correction instructions for a set of atomic relationships using the atomic janitor agent.

        Parameters:
            input_relationships (List[ArchitectAgentRelationship]): The relationships (units of work) to validate.
            text (str): Human-readable context or explanation to guide validation.
            targeting (Optional[Node]): Optional node context providing name, description, and properties to narrow validation.
            brain_id (str): Identifier of the brain/knowledge context to run the agent against.
            timeout (int): Maximum seconds to wait for a single agent invocation before raising a timeout.
            max_retries (int): Maximum number of retry attempts for agent invocation on timeout.

        Returns:
            AtomicJanitorAgentInputOutput | str: `'OK'` if all relationships are valid; otherwise an AtomicJanitorAgentInputOutput containing instructions or corrections.

        Raises:
            TimeoutError: If the agent invocation times out or fails after the configured retry attempts.
        """

        self._get_agent(
            output_schemas=AtomicJanitorAgentInputOutput,
            brain_id=brain_id,
            type_="atomic-janitor",
        )

        accumulated_messages = []

        def _invoke_agent(previous_messages: list = None):
            """
            Assemble a trimmed message history, append an atomic-janitor user message built from the current units of work, text, and optional targeting, then invoke the agent.

            Parameters:
                previous_messages (list | None): Prior messages to include in the invocation; if the list exceeds HISTORY_MAX_MESSAGES, older messages are removed via RemoveMessage objects before sending.

            Returns:
                The raw response returned by self.agent.invoke for the assembled messages.
            """
            messages_list = self._content_only_history(
                previous_messages, keep_last=HISTORY_MAX_MESSAGES_DELETE
            )
            messages_list.append(
                {
                    "role": "user",
                    "content": ATOMIC_JANITOR_AGENT_PROMPT.format(
                        units_of_work=json.dumps(
                            strip_properties(
                                [
                                    rel.model_dump(mode="json")
                                    for rel in input_relationships
                                ]
                            )
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
                    ),
                }
            )

            return self.agent.invoke(
                {"messages": messages_list},
                config={
                    "recursion_limit": 100,
                    "tags": ["janitor_agent"],
                    "metadata": {"agent": "janitor_agent", "brain_id": brain_id},
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(previous_messages: list):
            """
            Invoke the agent using the provided message history, update token counters from any returned usage metadata, and return the agent's response.

            Parameters:
                previous_messages (list): Accumulated conversation messages to send as context to the agent.

            Returns:
                dict: The agent's response object; expected to include a "messages" list containing response message objects.

            Raises:
                TimeoutError: If the invocation exceeds the configured timeout or if retries are exhausted; the exception message contains timing or last-attempt error details.
            """
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
                                "janitor_agent",
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
            """
            Invoke the agent with retry logic, append any returned messages to the outer `accumulated_messages` list, and return the agent response.

            Parameters:
                previous_messages (list): Messages to include in the invocation request.

            Returns:
                response (dict): The agent's response object; may include a `messages` key with returned messages.

            Raises:
                TimeoutError: If the invocation exhausts retry attempts or a timeout occurs.
            """
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(self._content_only_history(messages))
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
        """
        Update the agent's cumulative token counters from a message's usage metadata.

        Parameters:
            usage_metadata (dict): A mapping containing token usage information. Recognized keys:
                - "input_tokens": number to add to `input_tokens`
                - "output_tokens": number to add to `output_tokens`
                - "input_token_details": dict with optional "cache_read" to add to `cached_tokens`
                - "output_token_details": dict with optional "reasoning" to add to `reasoning_tokens`
        """
        # Base counts
        self.input_tokens += usage_metadata.get("input_tokens", 0)
        self.output_tokens += usage_metadata.get("output_tokens", 0)

        # Input details (caching)
        input_details = usage_metadata.get("input_token_details", {})
        self.cached_tokens += input_details.get("cache_read", 0)

        # Output details (reasoning)
        output_details = usage_metadata.get("output_token_details", {})
        self.reasoning_tokens += output_details.get("reasoning", 0)
