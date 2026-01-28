"""
File: /architect_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Dict, List, Literal, Optional, Tuple
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import RemoveMessage
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
from src.utils.tokens import token_detail_from_token_counts

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
        self.used_entities_set = []
        self.ingestion_manager = ingestion_manager
        # self.database_desc = database_desc

    def _get_tools(
        self,
        text: Optional[str] = None,
        entities: Optional[Dict[str, ScoutEntity]] = None,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
    ) -> List[BaseTool]:

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
        Run the architect agent.
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
            from langchain_core.messages import HumanMessage

            messages_list = []
            if previous_messages:
                messages = previous_messages
                if len(messages) > 10:
                    delete_count = len(messages) - 5
                    delete_ids = [msg.id for msg in messages[:delete_count]]
                    messages_list.extend(
                        [RemoveMessage(id=msg_id) for msg_id in delete_ids]
                    )
                    messages_list.extend(messages[delete_count:])
                else:
                    messages_list.extend(messages)

            user_message = HumanMessage(
                content=ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT.format(
                    text=text,
                    entities=[entity.model_dump(mode="json") for entity in ent],
                    previously_created_relationships=(
                        f"""
                    Previously Created Relationships: {[rel.model_dump(mode="json") for rel in all_rels]}
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
                )
            )
            messages_list.append(user_message)

            response = self.agent.invoke({"messages": messages_list})
            return response

        def _process_response(
            response: dict,
            connected_entity_uuids: set,
            all_relationships: list,
            all_new_nodes: list,
            entities: List[ScoutEntity],
            seen_relationship_keys: set,
        ) -> set:
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
                            self.token_detail = token_detail_from_token_counts(
                                self.input_tokens,
                                self.output_tokens,
                                self.cached_tokens,
                                self.reasoning_tokens,
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
            try:
                response = _invoke_agent_with_retry(
                    unconnected_entities_list, previous_messages
                )
                messages = response.get("messages", [])
                if messages:
                    accumulated_messages.extend(messages)
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
    ) -> List[ArchitectAgentRelationship]:
        """
        Run the architect agent tooler.
        """

        entities_dict = {entity.uuid: entity for entity in entities}
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
                content=ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT.format(
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
                )
            )
            messages_list.append(user_message)

            return self.agent.invoke(
                {"messages": messages_list},
                config={"recursion_limit": MAX_RECURSION_LIMIT},
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
                    f"Architect agent invoke timed out after {timeout} seconds. "
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
                    f"Architect agent invoke failed after {last_attempt.attempt_number} attempts. "
                    f"Last error: {last_attempt.exception()}"
                ) from last_attempt.exception()
            except TimeoutError:
                raise

        _invoke_and_process(accumulated_messages)

        return list(self.relationships_set)

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
