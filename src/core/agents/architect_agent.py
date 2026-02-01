"""
File: /architect_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:44:06 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Dict, List, Literal, Optional, Tuple
from langchain.agents import create_agent
from langchain.tools import BaseTool
from pydantic import BaseModel
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.kg import Node, Predicate
from src.constants.prompts.architect_agent import (
    ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT,
    ARCHITECT_AGENT_TOOLER_SYSTEM_PROMPT,
    ARCHITECT_AGENT_SYSTEM_PROMPT,
    ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
)
from src.core.agents.scout_agent import ScoutEntity
from src.constants.agents import (
    _ArchitectAgentResponse,
    ArchitectAgentResponse,
    ArchitectAgentNew,
    ArchitectAgentRelationship,
)
from src.core.agents.tools.architect_agent.ArchitectAgentCheckUsedEntitiesTool import (
    ArchitectAgentCheckUsedEntitiesTool,
)
from src.core.agents.tools.architect_agent.ArchitectAgentCreateRelationshipTool import (
    ArchitectAgentCreateRelationshipTool,
)
from src.core.agents.tools.architect_agent.ArchitectAgentGetRemainingEntitiesToProcessTool import (
    ArchitectAgentGetRemainingEntitiesToProcessTool,
)
from src.core.agents.tools.architect_agent.ArchitectAgentMarkEntitiesAsUsedTool import (
    ArchitectAgentMarkEntitiesAsUsedTool,
)
from src.core.saving.ingestion_manager import IngestionManager
from src.utils.cleanup import strip_properties
from src.utils.tokens import merge_token_details, token_detail_from_token_counts

# from src.core.agents.tools.kg_agent import (
#     KGAgentSearchGraphTool,
# )

HISTORY_MAX_MESSAGES = 25
HISTORY_MAX_MESSAGES_DELETE = 8
MAX_RECURSION_LIMIT = 100


class ArchitectAgent:
    """
    Architect agent.
    """

    entities: Dict[str, ScoutEntity]

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache_adapter: CacheAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
        # database_desc: str,
        ingestion_manager: IngestionManager,
    ):
        """
        Initialize the ArchitectAgent with the required adapters and managers and prepare internal runtime state.

        Parameters:
            llm_adapter (LLMAdapter): Adapter for interacting with the language model.
            cache_adapter (CacheAdapter): Adapter used for cache reads/writes.
            kg (GraphAdapter): Graph knowledge-base adapter for entity and relationship operations.
            vector_store (VectorStoreAdapter): Adapter for vector storage and retrieval.
            embeddings (EmbeddingsAdapter): Adapter that produces vector embeddings for content.
            ingestion_manager (IngestionManager): Manager responsible for ingesting external data into the system.

        The constructor initializes internal tracking state including token counters (input, output, cached, reasoning),
        message/agent state, and containers for discovered relationships and used entities.
        """
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None
        self.token_detail = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.relationships_set: List[ArchitectAgentRelationship] = []
        self.used_entities_dict = {}
        self.ingestion_manager = ingestion_manager
        self.session_id: Optional[str] = None
        self.janitor_agent = None
        self._janitor_agent_brain_id = None
        # self.database_desc = database_desc

    def _get_tools(
        self,
        text: Optional[str] = None,
        entities: Optional[Dict[str, ScoutEntity]] = None,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
    ) -> List[BaseTool]:
        """
        Builds the set of tools the agent uses for relationship creation and entity tracking.

        Parameters:
            text (Optional[str]): Optional prompt or context text passed to the create-relationship tool.
            entities (Optional[Dict[str, ScoutEntity]]): Optional mapping of entity UUIDs to ScoutEntity instances for context.
            brain_id (str): Identifier for the knowledge brain/namespace to scope KG operations.
            targeting (Optional[Node]): Optional target node context to guide relationship creation.

        Returns:
            List[BaseTool]: A list containing:
                - ArchitectAgentCreateRelationshipTool configured with the provided context,
                - ArchitectAgentGetRemainingEntitiesToProcessTool,
                - ArchitectAgentCheckUsedEntitiesTool,
                - ArchitectAgentMarkEntitiesAsUsedTool.
        """
        return [
            ArchitectAgentCreateRelationshipTool(
                self,
                text=text,
                entities=entities,
                kg=self.kg,
                brain_id=brain_id,
                targeting=targeting,
            ),
            ArchitectAgentGetRemainingEntitiesToProcessTool(
                self,
            ),
            ArchitectAgentCheckUsedEntitiesTool(
                self,
            ),
            ArchitectAgentMarkEntitiesAsUsedTool(
                self,
            ),
        ]

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

    def _get_agent(
        self,
        type_: Literal["single", "tooler"],
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        text: Optional[str] = None,
        extra_system_prompt: Optional[dict] = None,
        entities: Optional[Dict[str, ScoutEntity]] = None,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
    ):
        """
        Configure and create the internal LangChain agent and store it on self.agent.

        Parameters:
            type_ (Literal["single", "tooler"]): Chooses agent mode. "single" uses a structured response prompt and no tools; "tooler" uses the tooler system prompt and enables tools.
            tools (Optional[List[BaseTool]]): Explicit tool list to attach when type_ is "tooler". If omitted for "tooler", a default tool set is created from provided context.
            output_schema (Optional[BaseModel]): Schema used as the agent's response format when type_ is "single".
            text (Optional[str]): Optional prompt context forwarded to default tool construction when tools are not provided.
            extra_system_prompt (Optional[dict]): Additional system prompt content interpolated into the selected system prompt.
            entities (Optional[Dict[str, ScoutEntity]]): Entity context forwarded to default tool construction when tools are not provided.
            brain_id (str): Identifier forwarded to default tool construction when tools are not provided.
            targeting (Optional[Node]): Targeting context forwarded to default tool construction when tools are not provided.

        Side effects:
            Creates an agent via create_agent and assigns it to self.agent.
        """
        if type_ == "single":
            system_prompt = ARCHITECT_AGENT_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        elif type_ == "tooler":
            system_prompt = ARCHITECT_AGENT_TOOLER_SYSTEM_PROMPT.format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(
                (
                    tools
                    if tools
                    else self._get_tools(
                        entities=entities,
                        brain_id=brain_id,
                        targeting=targeting,
                        text=text,
                    )
                )
                if type_ == "tooler"
                else []
            ),
            system_prompt=system_prompt,
            response_format=(
                (output_schema if output_schema else None)
                if type_ == "single"
                else None
            ),
            debug=os.environ.get("DEBUG", "false").lower() == "true",
        )

    def run(
        self,
        text: str,
        entities: List[ScoutEntity],
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> ArchitectAgentResponse:
        """
        Orchestrates the agent to discover relationships and new nodes for the provided entities based on the input text.

        Parameters:
            text (str): Natural-language description or instructions guiding relationship discovery.
            entities (List[ScoutEntity]): Entities to process; each entity should include a UUID.
            targeting (Optional[Node]): Optional node that provides contextual focus for relationship creation.
            brain_id (str): Identifier for the knowledge brain or workspace to use.
            timeout (int): Maximum seconds to wait for a single LLM invocation before timing out.
            max_retries (int): Number of retry attempts for timed-out LLM invocations.

        Returns:
            ArchitectAgentResponse: Contains:
                - new_nodes: list of newly discovered nodes produced by the agent.
                - relationships: list of relationships the agent created between entities or new nodes.
                - input_tokens: count of input tokens consumed during this run.
                - output_tokens: count of output tokens produced during this run.
        """

        entities_dict = {entity.uuid: entity for entity in entities}
        self.entities = entities_dict

        self._get_agent(
            output_schema=_ArchitectAgentResponse,
            brain_id=brain_id,
            targeting=targeting,
            type_="single",
        )

        self.input_tokens = 0
        self.output_tokens = 0

        def _invoke_agent(
            ent: list[ScoutEntity],
            all_rels: list[ArchitectAgentRelationship],
            previous_messages: list = None,
        ):
            """
            Builds a message history including the provided entities and previously created relationships, invokes the configured agent with that history, and returns the agent's response.

            Parameters:
                ent (list[ScoutEntity]): Entities to include in the prompt.
                all_rels (list[ArchitectAgentRelationship]): Previously created relationships to include in the prompt.
                previous_messages (list, optional): Prior message objects to include as conversation history; may be trimmed to fit history limits.

            Returns:
                The agent's response object containing the model's reply and associated metadata.
            """
            messages_list = self._content_only_history(previous_messages, keep_last=5)
            messages_list.append(
                {
                    "role": "user",
                    "content": ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT.format(
                        text=text,
                        entities=[entity.model_dump(mode="json") for entity in ent],
                        previously_created_relationships=(
                            f"""
                    Previously Created Relationships: {strip_properties([rel.model_dump(mode="json") for rel in all_rels])}
                    """
                            if len(all_rels) > 0
                            else ""
                        ),
                        targeting=(
                            f"""
                    The information is related to the following node:
                    Name: {targeting.name}
                    UUID: {targeting.uuid}
                    Type: {targeting.labels}
                    Description: {targeting.description}
                    {targeting.properties}
                    """
                            if targeting
                            else ""
                        ),
                    ),
                }
            )

            response = self.agent.invoke(
                {"messages": messages_list},
                config={
                    "tags": ["architect_agent"],
                    "metadata": {"agent": "architect_agent", "brain_id": brain_id},
                },
            )
            return response

        def _process_response(
            response: dict,
            connected_entity_uuids: set,
            all_relationships: list,
            all_new_nodes: list,
            entities: List[ScoutEntity],
            seen_relationship_keys: set,
        ) -> set:
            """
            Extracts newly created relationships and nodes from a structured agent response and updates the provided tracking collections.

            Parameters:
                response (dict): Agent response containing a `structured_response` with optional `relationships` and `new_nodes`.
                connected_entity_uuids (set): Set of UUIDs already known to be connected; will be updated with newly connected UUIDs.
                all_relationships (list): List to append newly discovered, deduplicated relationship objects.
                all_new_nodes (list): List to append any new node objects reported in the response.
                entities (List[ScoutEntity]): Source entities to check membership of relationship endpoints (used to mark endpoints as connected).
                seen_relationship_keys (set): Set of (tail_uuid, tip_uuid, relationship_name) tuples used to deduplicate relationships.

            Returns:
                set: The set of entity UUIDs that became connected as a result of processing this response iteration.
            """
            structured_response = response.get("structured_response", {})
            iteration_connected = set()

            if hasattr(structured_response, "relationships"):
                new_relationships = []
                for rel in structured_response.relationships:
                    if (
                        hasattr(rel, "tip")
                        and hasattr(rel.tip, "uuid")
                        and hasattr(rel, "tail")
                        and hasattr(rel.tail, "uuid")
                    ):
                        tip_uuid = rel.tip.uuid
                        tail_uuid = rel.tail.uuid
                        rel_key = (tail_uuid, tip_uuid, rel.name)
                        if rel_key not in seen_relationship_keys:
                            seen_relationship_keys.add(rel_key)
                            new_relationships.append(rel)
                            if any(e.uuid == tip_uuid for e in entities):
                                iteration_connected.add(tip_uuid)
                            if any(e.uuid == tail_uuid for e in entities):
                                iteration_connected.add(tail_uuid)

                if iteration_connected:
                    connected_entity_uuids.update(iteration_connected)
                    all_relationships.extend(new_relationships)

            if hasattr(structured_response, "new_nodes"):
                all_new_nodes.extend(structured_response.new_nodes)

            return iteration_connected

        connected_entity_uuids = set()
        all_relationships = []
        all_new_nodes = []
        seen_relationship_keys = set()
        max_iterations = 3
        iteration = 0
        accumulated_messages = []

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(
            unconnected_entities_list: List[ScoutEntity], previous_messages: list
        ):
            """
            Invoke the architect agent with a single-worker executor and timeout, and update token counts from any returned messages.

            Parameters:
                unconnected_entities_list (List[ScoutEntity]): Entities to include in the agent invocation.
                previous_messages (list): Message history to send to the agent.

            Returns:
                dict: The response dictionary returned by the agent invocation.

            Raises:
                TimeoutError: If the agent call does not complete within the configured timeout.
            """
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _invoke_agent,
                        unconnected_entities_list,
                        all_relationships,
                        previous_messages,
                    )
                    response = future.result(timeout=timeout)
                    for m in response.get("messages", []):
                        if hasattr(m, "usage_metadata"):
                            self._update_token_counts(m.usage_metadata)
                            self.token_detail = merge_token_details(
                                [
                                    self.token_detail,
                                    token_detail_from_token_counts(
                                        self.input_tokens,
                                        self.output_tokens,
                                        self.cached_tokens,
                                        self.reasoning_tokens,
                                        "architect_agent",
                                    ),
                                ]
                            )
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Architect agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(
            unconnected_entities_list: List[ScoutEntity], previous_messages: list
        ):
            """
            Invoke the agent for the provided unconnected entities, append any returned messages to the accumulated message history, and process the agent's response to determine which entities became connected.

            Parameters:
                unconnected_entities_list (List[ScoutEntity]): Entities to include in this agent invocation.
                previous_messages (list): Message history to send with the invocation.

            Returns:
                set: UUID strings of entities that were connected as a result of this invocation.

            Raises:
                TimeoutError: If the agent invocation exhausts retries or times out.
            """
            try:
                response = _invoke_agent_with_retry(
                    unconnected_entities_list, previous_messages
                )
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(self._content_only_history(messages))
                return _process_response(
                    response,
                    connected_entity_uuids,
                    all_relationships,
                    all_new_nodes,
                    entities,
                    seen_relationship_keys,
                )
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Architect agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        unconnected_entities = [
            entity for entity in entities if entity.uuid not in connected_entity_uuids
        ]

        while len(unconnected_entities) > 0 and iteration < max_iterations:
            ret = _invoke_and_process(unconnected_entities, accumulated_messages)
            unconnected_entities = list(
                filter(lambda e: e.uuid not in ret, unconnected_entities)
            )
            iteration += 1

        if not all_relationships and not all_new_nodes:
            return ArchitectAgentResponse(
                new_nodes=[],
                relationships=[],
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )

        return ArchitectAgentResponse(
            new_nodes=[
                ArchitectAgentNew(**new_node.model_dump(mode="json"))
                for new_node in all_new_nodes
            ],
            relationships=[
                ArchitectAgentRelationship(**relationship.model_dump(mode="json"))
                for relationship in all_relationships
            ],
        )

    def run_tooler(
        self,
        text: str,
        entities: List[ScoutEntity],
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 3600,
        max_retries: int = 3,
        ingestion_session_id: Optional[str] = None,
    ) -> List[ArchitectAgentRelationship]:
        """
        Drive the architect agent in "tooler" mode to iteratively discover relationships using available tools and collect the results.

        This invokes the agent with the provided text and entities, manages message history trimming, updates token accounting from message metadata, and accumulates relationships produced by tool-driven agent actions.

        Parameters:
            text (str): Natural-language prompt or instructions for relationship discovery.
            entities (List[ScoutEntity]): Candidate entities the agent may connect; each entity must include a UUID.
            targeting (Optional[Node]): Optional node to which discovered information should be anchored or related.
            brain_id (str): Identifier for the knowledge brain/context to use.
            timeout (int): Maximum seconds to wait for a single agent invocation before raising a timeout.
            max_retries (int): Maximum number of retry attempts for timed or retried invocations.

        Returns:
            List[ArchitectAgentRelationship]: The relationships discovered and collected by the agent during this run.

        Raises:
            TimeoutError: If the agent fails to produce a response within `timeout` after the allowed retry attempts.
        """
        import uuid
        from src.lib.redis.client import _redis_client

        self.session_id = str(uuid.uuid4())
        self.relationships_set.clear()

        entities_dict = {
            entity.uuid: strip_properties([entity.model_dump(mode="json")])[0]
            for entity in entities
        }
        self.entities = entities_dict

        self._get_agent(
            type_="tooler",
            text=text,
            brain_id=brain_id,
            entities=entities_dict,
            targeting=targeting,
        )

        accumulated_messages = []

        def _invoke_agent(previous_messages: list = None):
            """
            Prepare a pruned message history, append a formatted human prompt for the "tooler" flow, and invoke the agent.

            Parameters:
                previous_messages (list | None): Prior messages to include in the history; if the count exceeds HISTORY_MAX_MESSAGES,
                    the oldest messages are removed via RemoveMessage entries and the remaining messages are preserved.

            Returns:
                The response object returned by self.agent.invoke when called with the constructed message list.
            """
            messages_list = self._content_only_history(
                previous_messages, keep_last=HISTORY_MAX_MESSAGES_DELETE
            )
            messages_list.append(
                {
                    "role": "user",
                    "content": ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT.format(
                        text=text,
                        targeting=(
                            f"""
                    The information is related to the following node:
                    Name: {targeting.name}
                    UUID: {targeting.uuid}
                    Type: {targeting.labels}
                    Description: {targeting.description}
                    {targeting.properties}
                    """
                            if targeting
                            else ""
                        ),
                    ),
                }
            )

            metadata = {
                "agent": "architect_agent",
                "brain_id": brain_id,
            }
            if ingestion_session_id:
                metadata["ingestion_session_id"] = ingestion_session_id
            return self.agent.invoke(
                {"messages": messages_list},
                config={
                    "recursion_limit": MAX_RECURSION_LIMIT,
                    "tags": ["architect_agent", "architect_tooler"],
                    "metadata": metadata,
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
            Invoke the agent in a worker thread, enforce the configured timeout, and update token counters from any returned message usage metadata.

            Parameters:
                previous_messages (list): Message history to pass to the agent invocation.

            Returns:
                dict: The agent response object as returned by the underlying _invoke_agent call.

            Raises:
                TimeoutError: If the agent invocation exceeds the configured timeout.
            """
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent, previous_messages)
                    response = future.result(timeout=timeout)
                    for m in response.get("messages", []):
                        if hasattr(m, "usage_metadata"):
                            self._update_token_counts(m.usage_metadata)
                            self.token_detail = merge_token_details(
                                [
                                    self.token_detail,
                                    token_detail_from_token_counts(
                                        self.input_tokens,
                                        self.output_tokens,
                                        self.cached_tokens,
                                        self.reasoning_tokens,
                                        "architect_agent",
                                    ),
                                ]
                            )
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Architect agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(previous_messages: list):
            """
            Invoke the agent with retry logic and append any returned messages to the outer `accumulated_messages` list.

            Parameters:
                previous_messages (list): Message history to send to the agent.

            Returns:
                response (dict): The agent's response object; may include a "messages" key containing new messages.

            Raises:
                TimeoutError: If the agent invocation times out or fails after all retry attempts.
            """
            try:
                response = _invoke_agent_with_retry(previous_messages)
                messages = response.get("messages", [])
                if messages:
                    response_history = self._content_only_history(messages)
                    prior_history = self._content_only_history(previous_messages)
                    if (
                        prior_history
                        and response_history[: len(prior_history)] == prior_history
                    ):
                        response_history = response_history[len(prior_history) :]
                    if response_history:
                        accumulated_messages.extend(response_history)
                return response
            except RetryError as e:
                last_attempt = e.last_attempt
                raise TimeoutError(
                    f"Architect agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        _invoke_and_process(accumulated_messages)

        return list(self.relationships_set)

    def _update_token_counts(self, usage_metadata: dict):
        """
        Update the agent's token counters from an LLM usage metadata dictionary.

        Parameters:
            usage_metadata (dict): Metadata containing token usage details. Expected keys:
                - "input_tokens": integer count of input tokens.
                - "output_tokens": integer count of output tokens.
                - "input_token_details": dict with "cache_read" (int) for cached input token reads.
                - "output_token_details": dict with "reasoning" (int) for tokens spent on reasoning.
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
