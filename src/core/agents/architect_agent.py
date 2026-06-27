"""
File: /architect_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday April 12th 2026 1:35:36 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, Dict, List, Literal, Optional, Tuple

from langchain.tools import BaseTool
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.adapters.cache import CacheAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.config import config
from src.constants.agents import (
    ArchitectAgentEntity,
    ArchitectAgentNew,
    ArchitectAgentRelationship,
    ArchitectAgentResponse,
    _ArchitectAgentRelationship,
    _ArchitectAgentResponse,
)
from src.constants.kg import Node, Predicate
from src.constants.prompts.architect_agent import (
    ARCHITECT_AGENT_COARSE_TOOLER_CREATE_RELATIONSHIPS_PROMPT,
    ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
    ARCHITECT_AGENT_SYSTEM_PROMPT,
    ARCHITECT_AGENT_TOOLER_COARSE_SYSTEM_PROMPT,
    ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT,
    ARCHITECT_AGENT_TOOLER_SYSTEM_PROMPT,
    STRUCTURED_ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
    STRUCTURED_ARCHITECT_AGENT_FIX_RELATIONSHIPS_PROMPT,
)
from src.core.agents.core import runtime_agent_factory
from src.core.agents.scout_agent import ScoutEntity
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
from src.core.plugins.prompts import prompt_registry
from src.core.saving.ingestion_manager import IngestionManager
from src.services.api.constants.requests import IngestionTripleSet
from src.utils.cleanup import strip_properties
from src.utils.similarity.vectors import cosine_similarity
from src.utils.tokens import merge_token_details, token_detail_from_token_counts

# from src.core.agents.tools.kg_agent import (
#     KGAgentSearchGraphTool,
# )

HISTORY_MAX_MESSAGES = 25
HISTORY_MAX_MESSAGES_DELETE = 8
MAX_RECURSION_LIMIT = 100


def _ingestion_partial_node_entity(node) -> ArchitectAgentEntity:
    return ArchitectAgentEntity(
        uuid=node.uuid or str(uuid.uuid4()),
        name=node.name,
        type=node.type,
        description=node.description,
        properties=node.properties or {},
    )


def _to_architect_relationship(relationship) -> ArchitectAgentRelationship:
    if isinstance(relationship, ArchitectAgentRelationship):
        return relationship
    data = (
        dict(relationship)
        if isinstance(relationship, dict)
        else relationship.model_dump(mode="json")
    )
    if not data.get("flow_key"):
        data["flow_key"] = str(uuid.uuid4())
    return ArchitectAgentRelationship(**data)


class ArchitectAgent:
    """
    Architect agent.
    """

    _SYSTEM_PROMPT_BUILDERS: dict[tuple[str, str], Callable[[str], str]] = {
        ("single", "granular"): lambda extra: prompt_registry.get(
            "ARCHITECT_AGENT_SYSTEM_PROMPT", ARCHITECT_AGENT_SYSTEM_PROMPT
        ).format(extra_system_prompt=extra),
        ("single", "coarse"): lambda extra: prompt_registry.get(
            "ARCHITECT_AGENT_SYSTEM_PROMPT", ARCHITECT_AGENT_SYSTEM_PROMPT
        ).format(extra_system_prompt=extra),
        ("tooler", "granular"): lambda extra: prompt_registry.get(
            "ARCHITECT_AGENT_TOOLER_SYSTEM_PROMPT", ARCHITECT_AGENT_TOOLER_SYSTEM_PROMPT
        ).format(extra_system_prompt=extra),
        ("tooler", "coarse"): lambda extra: prompt_registry.get(
            "ARCHITECT_AGENT_TOOLER_COARSE_SYSTEM_PROMPT",
            ARCHITECT_AGENT_TOOLER_COARSE_SYSTEM_PROMPT,
        ).format(extra_system_prompt=extra),
    }

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
        mode: Literal["granular", "coarse"] = "granular",
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
                mode=mode,
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
        mode: Literal["granular", "coarse"] = "granular",
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

        system_prompt = self._resolve_system_prompt(type_, mode, extra_system_prompt)
        tools = (
            (
                tools
                if tools
                else self._get_tools(
                    entities=entities,
                    brain_id=brain_id,
                    targeting=targeting,
                    text=text,
                    mode=mode,
                )
            )
            if type_ == "tooler"
            else []
        )
        response_format = (
            (output_schema if output_schema else None) if type_ == "single" else None
        )

        self.agent = runtime_agent_factory.build(
            model=self.llm_adapter.llm.langchain_model,
            tools=tools,
            system_prompt=system_prompt,
            output_schema=response_format,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            architecture=config.agentic_architecture,
            use_custom_backend=(mode == "coarse"),
        )

    def _resolve_system_prompt(
        self,
        type_: Literal["single", "tooler"],
        mode: Literal["granular", "coarse"],
        extra_system_prompt: Optional[dict] = None,
    ) -> str:
        resolved_extra_system_prompt = (
            extra_system_prompt if extra_system_prompt else ""
        )
        if mode not in ("granular", "coarse"):
            raise ValueError(f"Invalid mode for architect agent: {mode}")
        if type_ not in ("single", "tooler"):
            raise ValueError(f"Invalid type for architect agent: {type_}")
        key = (type_, mode)
        if key not in self._SYSTEM_PROMPT_BUILDERS:
            raise ValueError(
                f"Unsupported architect configuration: type={type_}, mode={mode}"
            )
        return self._SYSTEM_PROMPT_BUILDERS[key](resolved_extra_system_prompt)

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
                    "content": prompt_registry.get(
                        "ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT",
                        ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
                    ).format(
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
                _to_architect_relationship(relationship)
                for relationship in all_relationships
            ],
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def _persist_relationships(
        self,
        relationships: List[ArchitectAgentRelationship],
        text: str,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
        mode: Literal["granular", "coarse"] = "granular",
    ) -> Tuple[list, List[ScoutEntity]]:
        if not relationships:
            return [], []

        if not self.session_id:
            self.session_id = str(uuid.uuid4())

        rel_key = str(uuid.uuid4())
        input_rels = [
            _ArchitectAgentRelationship(
                tip=rel.tip,
                tail=rel.tail,
                name=rel.name,
                description=rel.description,
                properties=getattr(rel, "properties", {}) or {},
                **({"amount": rel.amount} if getattr(rel, "amount", None) else {}),
            )
            for rel in relationships
        ]
        output_rels: List[ArchitectAgentRelationship] = []
        fixed_relationships = []
        fixed_rels_sets = set()
        janitor_response = None
        janitor_new_entities: List[ScoutEntity] = []

        if mode == "granular":
            from src.core.agents.janitor_agent import JanitorAgent
            from src.lib.neo4j.client import _neo4j_client
            from src.services.input.agents import (
                embeddings_adapter,
                graph_adapter,
                llm_small_adapter,
                vector_store_adapter,
            )

            janitor_agent = self.janitor_agent
            if janitor_agent is None or self._janitor_agent_brain_id != brain_id:
                janitor_agent = JanitorAgent(
                    llm_small_adapter,
                    kg=graph_adapter,
                    vector_store=vector_store_adapter,
                    embeddings=embeddings_adapter,
                    database_desc=_neo4j_client.graphdb_description,
                )
                self.janitor_agent = janitor_agent
                self._janitor_agent_brain_id = brain_id
            start_input_tokens = janitor_agent.input_tokens
            start_output_tokens = janitor_agent.output_tokens
            start_cached_tokens = janitor_agent.cached_tokens
            start_reasoning_tokens = janitor_agent.reasoning_tokens

            janitor_response = janitor_agent.run_atomic_janitor(
                input_relationships=input_rels,
                text=text,
                targeting=targeting,
                brain_id=brain_id,
                timeout=300,
                max_retries=3,
            )

            janitor_token_detail = token_detail_from_token_counts(
                janitor_agent.input_tokens - start_input_tokens,
                janitor_agent.output_tokens - start_output_tokens,
                janitor_agent.cached_tokens - start_cached_tokens,
                janitor_agent.reasoning_tokens - start_reasoning_tokens,
                "janitor_agent",
            )
            self.token_detail = merge_token_details(
                [self.token_detail, janitor_token_detail]
            )

            required_new_nodes = getattr(janitor_response, "required_new_nodes", [])
            if required_new_nodes:
                for node in required_new_nodes:
                    scout_entity = ScoutEntity(
                        uuid=node.uuid,
                        name=node.name,
                        type=node.type,
                        description=node.description,
                        properties=node.properties,
                    )
                    self.entities[scout_entity.uuid] = scout_entity
                    janitor_new_entities.append(scout_entity)

            fixed_relationships = (
                getattr(janitor_response, "fixed_relationships", []) or []
            )
            if fixed_relationships:
                fixed_rels_sets = set(
                    frozenset((fr.tip.uuid, fr.tail.uuid, fr.name))
                    for fr in fixed_relationships
                )
                output_rels.extend(
                    [
                        ArchitectAgentRelationship(
                            flow_key=rel_key,
                            tip=rel.tip,
                            name=rel.name,
                            description=rel.description,
                            tail=rel.tail,
                            properties=getattr(rel, "properties", {}),
                            **(
                                {"amount": getattr(rel, "amount", None)}
                                if getattr(rel, "amount", None)
                                else {}
                            ),
                        )
                        for rel in fixed_relationships
                    ]
                )

        from src.services.input.agents import embeddings_small_adapter

        texts_to_embed = set()
        for rel in input_rels:
            texts_to_embed.add(rel.description if rel.description else rel.name)
        for rel in fixed_relationships:
            texts_to_embed.add(rel.description if rel.description else rel.name)

        text_to_embedding = {}
        if texts_to_embed:
            texts_list = list(texts_to_embed)
            vectors = embeddings_small_adapter.embed_texts(texts_list)
            for embed_text, vector in zip(texts_list, vectors):
                text_to_embedding[embed_text] = vector.embeddings

        for rel in input_rels:
            have_similar_relation = False
            if frozenset((rel.tip.uuid, rel.tail.uuid, rel.name)) in fixed_rels_sets:
                have_similar_relation = True
            else:
                rels_with_same_subject_and_object = [
                    fr
                    for fr in fixed_relationships
                    if (fr.tip.uuid == rel.tip.uuid and fr.tail.uuid == rel.tail.uuid)
                    or (fr.tip.uuid == rel.tail.uuid and fr.tail.uuid == rel.tip.uuid)
                ]

                if rels_with_same_subject_and_object:
                    input_rel_text = rel.description if rel.description else rel.name
                    input_embedding = text_to_embedding.get(input_rel_text)

                    candidates = []
                    for fr in rels_with_same_subject_and_object:
                        fixed_rel_text = fr.description if fr.description else fr.name
                        fixed_embedding = text_to_embedding.get(fixed_rel_text)

                        if input_embedding and fixed_embedding:
                            candidates.append(
                                (
                                    cosine_similarity(fixed_embedding, input_embedding),
                                    fr,
                                )
                            )

                    if candidates:
                        similarity_score, _ = max(candidates, key=lambda x: x[0])
                        if similarity_score > 0.90:
                            have_similar_relation = True

            if not have_similar_relation:
                source_rel = next(
                    (
                        r
                        for r in relationships
                        if r.tip.uuid == rel.tip.uuid
                        and r.tail.uuid == rel.tail.uuid
                        and r.name == rel.name
                    ),
                    None,
                )
                output_rels.append(
                    ArchitectAgentRelationship(
                        uuid=getattr(source_rel, "uuid", None) or str(uuid.uuid4()),
                        flow_key=getattr(source_rel, "flow_key", None) or rel_key,
                        tip=rel.tip,
                        name=rel.name,
                        description=rel.description,
                        tail=rel.tail,
                        properties=getattr(rel, "properties", {}),
                        **(
                            {"amount": getattr(rel, "amount", None)}
                            if getattr(rel, "amount", None)
                            else {}
                        ),
                    )
                )

        relationships_data = [
            rel.model_dump(mode="json")
            for rel in output_rels
            if isinstance(rel, ArchitectAgentRelationship)
        ]
        if relationships_data:
            from src.lib.redis.client import _redis_client
            from src.workers.tasks.ingestion import process_architect_relationships

            if self.session_id:
                _redis_client.client.incr(
                    f"{brain_id}:session:{self.session_id}:pending_tasks"
                )

            process_architect_relationships.delay(
                {
                    "relationships": relationships_data,
                    "brain_id": brain_id,
                    "session_id": self.session_id,
                }
            )

        self.relationships_set.extend(output_rels)

        if self.session_id and output_rels:
            from src.lib.redis.client import _redis_client

            _redis_client.set(
                f"session:{self.session_id}:relationships",
                json.dumps(
                    [rel.model_dump(mode="json") for rel in self.relationships_set]
                ),
                brain_id=brain_id,
                expires_in=3600,
            )

        wrong_relationships = (
            getattr(janitor_response, "wrong_relationships", []) or []
        )
        return wrong_relationships, janitor_new_entities

    def run_structured(
        self,
        text: str,
        entities: List[ScoutEntity],
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
        ingestion_session_id: Optional[str] = None,
        partial_triples: List[IngestionTripleSet] = [],
        current_triples: List[IngestionTripleSet] = [],
    ) -> ArchitectAgentResponse:
        """
        Given the provided text and partial/full triples, discover relationships and new nodes for the provided entities.

        Parameters:
            text (str): Natural-language description or instructions guiding relationship discovery.
            entities (List[ScoutEntity]): Entities to process; each entity should include a UUID.
            targeting (Optional[Node]): Optional node that provides contextual focus for relationship creation.
            brain_id (str): Identifier for the knowledge brain or workspace to use.
            timeout (int): Maximum seconds to wait for a single LLM invocation before timing out.
            max_retries (int): Number of retry attempts for timed-out LLM invocations.
            partial_triples (List[IngestionTripleSet]): Partial triples provided to use for the composition of the final triples.
            current_triples (List[IngestionTripleSet]): Current triples provided to be used in the prompt preventing duplicates.

        Returns:
            ArchitectAgentResponse: Contains:
                - new_nodes: list of newly discovered nodes produced by the agent.
                - relationships: list of relationships the agent created between entities or new nodes.
                - input_tokens: count of input tokens consumed during this run.
                - output_tokens: count of output tokens produced during this run.
        """

        triple_entity_registry: Dict[Tuple[str, str], ArchitectAgentEntity] = {}

        def _triple_entity(node) -> ArchitectAgentEntity:
            key = (node.name.strip().lower(), (node.type or "").strip().lower())
            if key not in triple_entity_registry:
                triple_entity_registry[key] = _ingestion_partial_node_entity(node)
            return triple_entity_registry[key]

        triple_relationships: List[ArchitectAgentRelationship] = []
        for cr in current_triples:
            if not cr.subject or not cr.subj_event:
                continue
            flow_key = str(uuid.uuid4())
            subject = _triple_entity(cr.subject)
            event = _triple_entity(cr.event)
            obj = _triple_entity(cr.object)
            triple_relationships.extend(
                [
                    ArchitectAgentRelationship(
                        tail=subject,
                        name=cr.subj_event.name,
                        tip=event,
                        description=cr.subj_event.description,
                        amount=cr.subj_event.amount,
                        properties=cr.subj_event.properties or {},
                        uuid=str(uuid.uuid4()),
                        flow_key=flow_key,
                    ),
                    ArchitectAgentRelationship(
                        tail=event,
                        name=cr.event_obj.name,
                        tip=obj,
                        description=cr.event_obj.description,
                        amount=cr.event_obj.amount,
                        properties=cr.event_obj.properties or {},
                        uuid=str(uuid.uuid4()),
                        flow_key=flow_key,
                    ),
                ]
            )
        for pt in partial_triples:
            flow_key = str(uuid.uuid4())
            event = _triple_entity(pt.event)
            obj = _triple_entity(pt.object)
            triple_relationships.append(
                ArchitectAgentRelationship(
                    tail=event,
                    name=pt.event_obj.name,
                    tip=obj,
                    description=pt.event_obj.description,
                    amount=pt.event_obj.amount,
                    properties=pt.event_obj.properties or {},
                    uuid=str(uuid.uuid4()),
                    flow_key=flow_key,
                ),
            )

        triple_keys = set(triple_entity_registry.keys())
        entities = [
            *[
                ScoutEntity(
                    uuid=te.uuid,
                    name=te.name,
                    type=te.type,
                    description=te.description,
                    properties=te.properties or {},
                )
                for te in triple_entity_registry.values()
            ],
            *[
                e
                for e in entities
                if (e.name.strip().lower(), (e.type or "").strip().lower())
                not in triple_keys
            ],
        ]

        entities_dict = {entity.uuid: entity for entity in entities}
        self.entities = entities_dict
        self.session_id = str(uuid.uuid4())
        self.relationships_set.clear()

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
            fix_content: Optional[str] = None,
        ):
            """
            Builds a message history including the provided entities and previously created relationships, invokes the configured agent with that history, and returns the agent's response.

            Parameters:
                ent (list[ScoutEntity]): Entities to include in the prompt.
                all_rels (list[ArchitectAgentRelationship]): Previously created relationships to include in the prompt.
                previous_messages (list, optional): Prior message objects to include as conversation history; may be trimmed to fit history limits.
                fix_content (Optional[str]): When provided, replaces the standard creation prompt with a correction prompt built from janitor feedback.

            Returns:
                The agent's response object containing the model's reply and associated metadata.
            """
            messages_list = self._content_only_history(previous_messages, keep_last=5)
            prompt_relationships = [*triple_relationships, *all_rels]

            messages_list.append(
                {
                    "role": "user",
                    "content": fix_content
                    if fix_content is not None
                    else prompt_registry.get(
                        "STRUCTURED_ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT",
                        STRUCTURED_ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
                    ).format(
                        text=text,
                        entities=[entity.model_dump(mode="json") for entity in ent],
                        previously_created_relationships=(
                            f"""
                    Previously Created Relationships: {strip_properties([rel.model_dump(mode="json") for rel in prompt_relationships])}
                    """
                            if len(prompt_relationships) > 0
                            else ""
                        ),
                        targeting=(
                            f"""
                    The information is related to the following node,
                    you must connect your relationships to this node directly by using it in the relationship "tail" or "tip" properties 
                    OR by creating new relationships that connect the relationships you just created to this node to create a unformed network:
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
            response = self.agent.invoke(
                {"messages": messages_list},
                config={
                    "tags": ["architect_agent"],
                    "metadata": metadata,
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
            unconnected_entities_list: List[ScoutEntity],
            previous_messages: list,
            fix_content: Optional[str] = None,
        ):
            """
            Invoke the architect agent with a single-worker executor and timeout, and update token counts from any returned messages.

            Parameters:
                unconnected_entities_list (List[ScoutEntity]): Entities to include in the agent invocation.
                previous_messages (list): Message history to send to the agent.
                fix_content (Optional[str]): Optional correction prompt that replaces the standard creation prompt.

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
                        fix_content,
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

        relationships_to_persist: List[ArchitectAgentRelationship] = []
        pending_batch = [
            _to_architect_relationship(relationship)
            for relationship in all_relationships
        ]
        max_janitor_iterations = 3
        janitor_iteration = 0

        while pending_batch:
            relationships_to_persist.extend(pending_batch)
            wrong_relationships, janitor_new_entities = self._persist_relationships(
                pending_batch,
                text=text,
                brain_id=brain_id,
                targeting=targeting,
            )
            pending_batch = []
            janitor_iteration += 1
            if not wrong_relationships or janitor_iteration >= max_janitor_iterations:
                break

            fix_content = prompt_registry.get(
                "STRUCTURED_ARCHITECT_AGENT_FIX_RELATIONSHIPS_PROMPT",
                STRUCTURED_ARCHITECT_AGENT_FIX_RELATIONSHIPS_PROMPT,
            ).format(
                wrong_relationships=strip_properties(
                    [wr.model_dump(mode="json") for wr in wrong_relationships]
                ),
                newly_created_nodes=(
                    "Newly created nodes available to use: "
                    + str(
                        strip_properties(
                            [e.model_dump(mode="json") for e in janitor_new_entities]
                        )
                    )
                    if janitor_new_entities
                    else ""
                ),
            )

            response = _invoke_agent_with_retry(
                [], accumulated_messages, fix_content=fix_content
            )
            messages = response.get("messages", [])
            if messages:
                accumulated_messages.extend(self._content_only_history(messages))

            structured_response = response.get("structured_response", {})
            if hasattr(structured_response, "new_nodes"):
                all_new_nodes.extend(structured_response.new_nodes)
            if hasattr(structured_response, "relationships"):
                for rel in structured_response.relationships:
                    if not (
                        hasattr(rel, "tip")
                        and hasattr(rel.tip, "uuid")
                        and hasattr(rel, "tail")
                        and hasattr(rel.tail, "uuid")
                    ):
                        continue
                    rel_key = (rel.tail.uuid, rel.tip.uuid, rel.name)
                    if rel_key in seen_relationship_keys:
                        continue
                    seen_relationship_keys.add(rel_key)
                    all_relationships.append(rel)
                    pending_batch.append(_to_architect_relationship(rel))

        return ArchitectAgentResponse(
            new_nodes=[
                ArchitectAgentNew(**new_node.model_dump(mode="json"))
                for new_node in all_new_nodes
            ],
            relationships=relationships_to_persist,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
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
        mode: Literal["granular", "coarse"] = "granular",
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
            mode=mode,
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
            content = ""
            if mode == "granular":
                content = prompt_registry.get(
                    "ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT",
                    ARCHITECT_AGENT_TOOLER_CREATE_RELATIONSHIPS_PROMPT,
                ).format(
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
            if mode == "coarse":
                content = prompt_registry.get(
                    "ARCHITECT_AGENT_COARSE_TOOLER_CREATE_RELATIONSHIPS_PROMPT",
                    ARCHITECT_AGENT_COARSE_TOOLER_CREATE_RELATIONSHIPS_PROMPT,
                ).format(
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
            messages_list.append(
                {
                    "role": "user",
                    "content": content,
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
