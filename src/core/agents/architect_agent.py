from src.core.saving.pipeline_runtime import routing_stage
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
from src.core.saving.ingest_cost import submit_with_context
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
    BATCH_ARCHITECT_EXTRACT_PROMPT,
    BATCH_ARCHITECT_REPAIR_PROMPT,
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
from src.core.saving.identity import (
    stable_flow_key,
    stable_node_id,
    stable_relationship_id,
)
from src.core.saving.ingestion_manager import IngestionManager
from src.core.search.catalog_graph import is_static_has_triple
from src.services.api.constants.requests import IngestionTripleSet
from src.utils.cleanup import strip_properties
from src.utils.dates import normalize_date_string
from src.utils.similarity.vectors import cosine_similarity

# from src.core.agents.tools.kg_agent import (
#     KGAgentSearchGraphTool,
# )

class ArchitectOutputLimitError(ValueError):
    """A relationship batch was cut off before its output could be validated."""


HISTORY_MAX_MESSAGES = 25
HISTORY_MAX_MESSAGES_DELETE = 8
MAX_RECURSION_LIMIT = 100


def _ingestion_partial_node_entity(node) -> ArchitectAgentEntity:
    return ArchitectAgentEntity(
        uuid=stable_node_id(
            node.name,
            node.type,
            getattr(node, "happened_at", None),
            getattr(node, "uuid", None),
        ),
        name=node.name,
        type=node.type,
        description=node.description,
        properties=node.properties or {},
        happened_at=getattr(node, "happened_at", None),
        polarity=getattr(node, "polarity", None) or "neutral",
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
        data["flow_key"] = stable_flow_key(
            event_uuid=None,
            event_name=(data.get("tip") or {}).get("name")
            if isinstance(data.get("tip"), dict)
            else getattr(data.get("tip"), "name", None),
            event_type=(data.get("tip") or {}).get("type")
            if isinstance(data.get("tip"), dict)
            else getattr(data.get("tip"), "type", None),
            happened_at=(data.get("tip") or {}).get("happened_at")
            if isinstance(data.get("tip"), dict)
            else getattr(data.get("tip"), "happened_at", None),
        )
    # Entity endpoints must have string uuids for ArchitectAgentEntity.
    for end_key in ("tail", "tip"):
        end = data.get(end_key)
        if isinstance(end, dict) and not end.get("uuid"):
            end = dict(end)
            end["uuid"] = str(uuid.uuid4())
            data[end_key] = end
    if not data.get("uuid"):
        tail = data.get("tail") or {}
        tip = data.get("tip") or {}
        tail_uuid = (
            tail.get("uuid")
            if isinstance(tail, dict)
            else getattr(tail, "uuid", "")
        )
        tip_uuid = (
            tip.get("uuid") if isinstance(tip, dict) else getattr(tip, "uuid", "")
        )
        data["uuid"] = stable_relationship_id(
            tail_uuid or "",
            data.get("name") or "",
            tip_uuid or "",
            data.get("flow_key"),
        )
    return ArchitectAgentRelationship(**data)


def ingestion_triples_to_relationships(
    current_triples: List[IngestionTripleSet],
    partial_triples: List[IngestionTripleSet],
) -> Tuple[List[ArchitectAgentRelationship], Dict[Tuple, ArchitectAgentEntity]]:
    triple_entity_registry: Dict[Tuple, ArchitectAgentEntity] = {}

    def _registry_key(node) -> Tuple:
        name = (node.name or "").strip().lower()
        entity_type = (node.type or "").strip().lower()
        if entity_type == "event":
            return (
                name,
                entity_type,
                normalize_date_string(getattr(node, "happened_at", None)) or "",
            )
        return (name, entity_type)

    def _triple_entity(node) -> ArchitectAgentEntity:
        key = _registry_key(node)
        if key not in triple_entity_registry:
            triple_entity_registry[key] = _ingestion_partial_node_entity(node)
        elif getattr(node, "uuid", None):
            existing = triple_entity_registry[key]
            if node.uuid and existing.uuid != node.uuid:
                triple_entity_registry[key] = _ingestion_partial_node_entity(node)
        return triple_entity_registry[key]

    triple_relationships: List[ArchitectAgentRelationship] = []
    for cr in current_triples:
        if not cr.subject or not cr.subj_event:
            continue
        subject = _triple_entity(cr.subject)
        obj = _triple_entity(cr.object)
        if cr.event is None or is_static_has_triple(cr):
            pred_name = cr.subj_event.name or "HAS"
            rel_uuid = cr.subj_event.uuid or stable_relationship_id(
                subject.uuid, pred_name, obj.uuid
            )
            triple_relationships.append(
                ArchitectAgentRelationship(
                    tail=subject,
                    name=pred_name,
                    tip=obj,
                    description=cr.subj_event.description,
                    amount=cr.subj_event.amount,
                    properties=cr.subj_event.properties or {},
                    uuid=rel_uuid,
                    flow_key=rel_uuid,
                )
            )
            continue
        if not cr.event or not cr.event_obj:
            continue
        event = _triple_entity(cr.event)
        flow_key = stable_flow_key(
            event_uuid=event.uuid,
            event_name=event.name,
            event_type=event.type,
            happened_at=event.happened_at,
            supplied=cr.subj_event.uuid or cr.event_obj.uuid,
        )
        subj_event_uuid = cr.subj_event.uuid or stable_relationship_id(
            subject.uuid, cr.subj_event.name, event.uuid, flow_key
        )
        event_obj_uuid = cr.event_obj.uuid or stable_relationship_id(
            event.uuid, cr.event_obj.name, obj.uuid, flow_key
        )
        triple_relationships.extend(
            [
                ArchitectAgentRelationship(
                    tail=subject,
                    name=cr.subj_event.name,
                    tip=event,
                    description=cr.subj_event.description,
                    amount=cr.subj_event.amount,
                    properties=cr.subj_event.properties or {},
                    uuid=subj_event_uuid,
                    flow_key=flow_key,
                ),
                ArchitectAgentRelationship(
                    tail=event,
                    name=cr.event_obj.name,
                    tip=obj,
                    description=cr.event_obj.description,
                    amount=cr.event_obj.amount,
                    properties=cr.event_obj.properties or {},
                    uuid=event_obj_uuid,
                    flow_key=flow_key,
                ),
            ]
        )
    for pt in partial_triples:
        if not pt.event or not pt.event_obj:
            continue
        event = _triple_entity(pt.event)
        obj = _triple_entity(pt.object)
        if pt.subject:
            _triple_entity(pt.subject)
        flow_key = stable_flow_key(
            event_uuid=event.uuid,
            event_name=event.name,
            event_type=event.type,
            happened_at=event.happened_at,
            supplied=pt.event_obj.uuid,
        )
        event_obj_uuid = pt.event_obj.uuid or stable_relationship_id(
            event.uuid, pt.event_obj.name, obj.uuid, flow_key
        )
        triple_relationships.append(
            ArchitectAgentRelationship(
                tail=event,
                name=pt.event_obj.name,
                tip=obj,
                description=pt.event_obj.description,
                amount=pt.event_obj.amount,
                properties=pt.event_obj.properties or {},
                uuid=event_obj_uuid,
                flow_key=flow_key,
            ),
        )

    return triple_relationships, triple_entity_registry


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
        self.relationships_set: List[ArchitectAgentRelationship] = []
        self.pending_persistence_batches: List[List[ArchitectAgentRelationship]] = []
        self.used_entities_dict = {}
        self.ingestion_manager = ingestion_manager
        self.session_id: Optional[str] = None
        self.janitor_agent = None
        self._janitor_agent_brain_id = None
        self.defer_janitor = True
        # self.database_desc = database_desc

    def take_pending_relationships(self) -> List[ArchitectAgentRelationship]:
        pending: List[ArchitectAgentRelationship] = []
        for batch in self.pending_persistence_batches:
            pending.extend(batch)
        self.pending_persistence_batches.clear()
        if pending:
            return pending
        return list(self.relationships_set)

    def queue_relationships_for_persistence(
        self, relationships: List[ArchitectAgentRelationship]
    ) -> None:
        if not relationships:
            return
        self.pending_persistence_batches.append(list(relationships))
        self.relationships_set.extend(relationships)

    def run_batched_janitor(
        self,
        *,
        text: str,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
        batch_size: int = 20,
        cost_ledger=None,
    ) -> None:
        """
        Run Janitor once per batch over pending relationships (post-Architect),
        with cheap grounding triage so LLM Janitor stays exception-only.
        """
        from src.core.agents.janitor_agent import JanitorAgent
        from src.core.saving.grounding import triage_relationships_for_janitor
        from src.services.input.agents import (
            embeddings_adapter,
            graph_adapter,
            llm_small_adapter,
            vector_store_adapter,
        )

        pending = []
        for batch in self.pending_persistence_batches:
            pending.extend(batch)
        if not pending:
            pending = list(self.relationships_set)
        if not pending:
            return

        triage = triage_relationships_for_janitor(pending, text)
        if cost_ledger is not None:
            cost_ledger.janitor_skipped += len(triage.accept)
            cost_ledger.janitor_rejected += len(triage.reject)
            cost_ledger.janitor_ran += 0
            for decision in triage.reject:
                cost_ledger.janitor_drop_reasons.append(decision.reason)

        if triage.reject:
            print(
                "[DEBUG (run_batched_janitor)]: deterministic rejects "
                f"count={len(triage.reject)} "
                f"reasons={[d.reason for d in triage.reject]}"
            )

        cleaned: List[ArchitectAgentRelationship] = list(triage.accept)
        need = list(triage.ambiguous)
        if cost_ledger is not None:
            cost_ledger.janitor_ambiguous += len(need)
        if not need:
            self.pending_persistence_batches = [cleaned] if cleaned else []
            self.relationships_set = list(cleaned)
            return

        janitor_agent = self.janitor_agent
        if janitor_agent is None or self._janitor_agent_brain_id != brain_id:
            janitor_agent = JanitorAgent(
                llm_small_adapter,
                kg=graph_adapter,
                vector_store=vector_store_adapter,
                embeddings=embeddings_adapter,
                database_desc=graph_adapter.graphdb_description,
            )
            self.janitor_agent = janitor_agent
            self._janitor_agent_brain_id = brain_id

        size = max(1, int(batch_size or 20))
        max_llm_calls = max(0, int(getattr(config, "ingest_janitor_max_llm_calls", 2)))
        llm_calls = 0
        for i in range(0, len(need), size):
            if llm_calls >= max_llm_calls:
                remaining = len(need) - i
                print(
                    "[DEBUG (run_batched_janitor)]: janitor LLM budget exhausted "
                    f"max_calls={max_llm_calls}; dropping remaining ambiguous={remaining}"
                )
                if cost_ledger is not None:
                    cost_ledger.janitor_drop_reasons.append(
                        f"janitor_budget_exhausted:{remaining}"
                    )
                break
            batch = need[i : i + size]
            input_rels = [
                _ArchitectAgentRelationship(
                    tip=rel.tip,
                    tail=rel.tail,
                    name=rel.name,
                    description=rel.description,
                    properties=getattr(rel, "properties", {}) or {},
                    **(
                        {"amount": rel.amount}
                        if getattr(rel, "amount", None)
                        else {}
                    ),
                )
                for rel in batch
            ]
            response = janitor_agent.run_atomic_janitor(
                input_relationships=input_rels,
                text=text,
                targeting=targeting,
                brain_id=brain_id,
                timeout=300,
                max_retries=3,
            )
            llm_calls += 1
            if cost_ledger is not None:
                cost_ledger.janitor_ran += 1

            # Parse / provider failure must not silently approve edges.
            if response is None:
                print(
                    "[DEBUG (run_batched_janitor)]: janitor parse/provider failure; "
                    f"dropping ambiguous batch size={len(batch)}"
                )
                if cost_ledger is not None:
                    cost_ledger.janitor_drop_reasons.append("janitor_parse_failure")
                continue

            if response == "OK":
                cleaned.extend(batch)
                continue

            status = getattr(response, "status", None)
            if status == "REJECT":
                veto = getattr(response, "veto_reasons", None) or []
                print(
                    "[DEBUG (run_batched_janitor)]: janitor veto "
                    f"count={len(batch)} reasons={veto}"
                )
                if cost_ledger is not None:
                    cost_ledger.janitor_drop_reasons.extend(
                        list(veto) or ["janitor_veto"]
                    )
                continue

            fixed = getattr(response, "fixed_relationships", None) or []
            wrong = getattr(response, "wrong_relationships", None) or []
            wrong_keys = set()
            for w in wrong:
                target = getattr(w, "relationship", None) or w
                wrong_keys.add(
                    (
                        getattr(getattr(target, "tail", None), "uuid", None),
                        getattr(getattr(target, "tip", None), "uuid", None),
                        getattr(target, "name", None),
                    )
                )
            if cost_ledger is not None and wrong:
                for w in wrong:
                    cost_ledger.janitor_drop_reasons.append(
                        getattr(w, "reason", None) or "janitor_wrong"
                    )
            for rel in batch:
                key = (rel.tail.uuid, rel.tip.uuid, rel.name)
                if key in wrong_keys:
                    continue
                cleaned.append(rel)
            for rel in fixed:
                cleaned.append(
                    ArchitectAgentRelationship(
                        flow_key=str(uuid.uuid4()),
                        tip=rel.tip,
                        tail=rel.tail,
                        name=rel.name,
                        description=rel.description,
                        properties=getattr(rel, "properties", {}) or {},
                        **(
                            {"amount": getattr(rel, "amount", None)}
                            if getattr(rel, "amount", None)
                            else {}
                        ),
                    )
                )

        self.pending_persistence_batches = [cleaned] if cleaned else []
        self.relationships_set = list(cleaned)

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

    @routing_stage("architect")
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
        """

        entities_dict = {entity.uuid: entity for entity in entities}
        self.entities = entities_dict

        self._get_agent(
            output_schema=_ArchitectAgentResponse,
            brain_id=brain_id,
            targeting=targeting,
            type_="single",
        )

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
                    future = submit_with_context(executor, 
                        _invoke_agent,
                        unconnected_entities_list,
                        all_relationships,
                        previous_messages,
                    )
                    response = future.result(timeout=timeout)
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
                    database_desc=graph_adapter.graphdb_description,
                )
                self.janitor_agent = janitor_agent
                self._janitor_agent_brain_id = brain_id
            janitor_response = janitor_agent.run_atomic_janitor(
                input_relationships=input_rels,
                text=text,
                targeting=targeting,
                brain_id=brain_id,
                timeout=300,
                max_retries=3,
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
        if output_rels:
            self.queue_relationships_for_persistence(output_rels)
        elif relationships_data:
            pass

        wrong_relationships = (
            getattr(janitor_response, "wrong_relationships", []) or []
        )
        return wrong_relationships, janitor_new_entities

    @routing_stage("architect")
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
        persist_submitted: bool = True,
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
        """

        triple_relationships, triple_entity_registry = ingestion_triples_to_relationships(
            current_triples, partial_triples
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
                    happened_at=te.happened_at,
                    polarity=te.polarity,
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
        self.pending_persistence_batches.clear()

        if triple_relationships and persist_submitted:
            self._persist_relationships(
                triple_relationships,
                text=text or "",
                brain_id=brain_id,
                targeting=targeting,
            )

        self._get_agent(
            output_schema=_ArchitectAgentResponse,
            brain_id=brain_id,
            targeting=targeting,
            type_="single",
        )


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
                    future = submit_with_context(executor, 
                        _invoke_agent,
                        unconnected_entities_list,
                        all_relationships,
                        previous_messages,
                        fix_content,
                    )
                    response = future.result(timeout=timeout)
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
        )

    @routing_stage("architect")
    def run_batch_extract(
        self,
        text: str,
        entities: List[ScoutEntity],
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 120,
        max_retries: int = 2,
        ingestion_session_id: Optional[str] = None,
        mode: Literal["granular", "coarse"] = "granular",
        reset: bool = True,
        cost_ledger=None,
        escalate: bool = True,
    ) -> List[ArchitectAgentRelationship]:
        """
        Pure tools=[] schema extract: one primary call, at most one repair, then
        optional escalate to run_tooler. No graph side effects before validation.
        """
        from src.core.agents.core.parsing import (
            get_finish_reason_from_response,
            parse_structured_from_messages,
        )
        from src.core.saving.architect_batch import (
            BatchExtractResponse,
            event_leg_incomplete,
            validate_batch_extract,
        )

        if reset or not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.relationships_set.clear()
            self.pending_persistence_batches.clear()

        entities_dict = {entity.uuid: entity for entity in entities}
        self.entities = {
            uuid_: strip_properties([entity.model_dump(mode="json")])[0]
            for uuid_, entity in entities_dict.items()
        }

        self._get_agent(
            output_schema=BatchExtractResponse,
            brain_id=brain_id,
            targeting=targeting,
            type_="single",
            mode=mode,
        )

        entity_payload = [
            strip_properties([entity.model_dump(mode="json")])[0]
            for entity in entities
        ]
        targeting_block = (
            f"""
The information is related to the following node; connect relationships to it
when relevant:
Name: {targeting.name}
UUID: {targeting.uuid}
Type: {targeting.labels}
Description: {targeting.description}
{targeting.properties}
"""
            if targeting
            else ""
        )

        schema_calls = 0
        repair_calls = 0
        max_schema_calls = max(1, int(config.ingest_architect_max_schema_calls))

        def _invoke(prompt: str) -> BatchExtractResponse:
            nonlocal schema_calls
            messages_list = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            metadata = {
                "agent": "architect_agent",
                "brain_id": brain_id,
                "loop_origin": "architect_batch",
            }
            if ingestion_session_id:
                metadata["ingestion_session_id"] = ingestion_session_id

            def _call():
                return self.agent.invoke(
                    {"messages": messages_list},
                    config={
                        "tags": ["architect_agent", "architect_batch"],
                        "metadata": metadata,
                    },
                )

            @retry(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(TimeoutError),
                reraise=True,
            )
            def _with_timeout():
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = submit_with_context(executor, _call)
                        return future.result(timeout=timeout)
                except FutureTimeoutError:
                    raise TimeoutError(
                        f"Architect batch extract timed out after {timeout}s"
                    )

            response = _with_timeout()
            schema_calls += 1
            messages = response.get("messages", [])
            finish_reason = (
                get_finish_reason_from_response(messages[-1]) if messages else None
            )
            if str(finish_reason).upper() in {"MAX_TOKENS", "LENGTH", "MAX_LENGTH"}:
                raise ArchitectOutputLimitError(
                    "Relationship extraction exceeded the model output limit."
                )
            structured = response.get("structured_response")
            if isinstance(structured, dict):
                try:
                    structured = BatchExtractResponse.model_validate(structured)
                except Exception:
                    structured = None
            if structured is None:
                fallback = parse_structured_from_messages(
                    response.get("messages", []), BatchExtractResponse
                )
                structured = fallback
            if structured is None:
                structured = BatchExtractResponse(relationships=[], new_nodes=[])
            return structured

        primary_prompt = prompt_registry.get(
            "BATCH_ARCHITECT_EXTRACT_PROMPT",
            BATCH_ARCHITECT_EXTRACT_PROMPT,
        ).format(
            text=text,
            entities=entity_payload,
            targeting=targeting_block,
        )
        extracted = _invoke(primary_prompt)
        validation = validate_batch_extract(
            extracted, source_text=text, scout_entities=entities
        )
        primary_validation = validation

        if validation.rejected and schema_calls < max_schema_calls:
            repair_prompt = prompt_registry.get(
                "BATCH_ARCHITECT_REPAIR_PROMPT",
                BATCH_ARCHITECT_REPAIR_PROMPT,
            ).format(
                errors="\n".join(validation.reasons),
                text=text,
                entities=entity_payload,
            )
            repair_calls += 1
            try:
                extracted = _invoke(repair_prompt)
            except ArchitectOutputLimitError:
                repaired_validation = primary_validation
            else:
                repaired_validation = validate_batch_extract(
                    extracted, source_text=text, scout_entities=entities
                )
            # Keep the more usable extract; do not discard a good primary
            # when repair returns fewer accepted edges.
            if len(repaired_validation.accepted) > len(primary_validation.accepted):
                validation = repaired_validation
            elif (
                len(repaired_validation.accepted) == len(primary_validation.accepted)
                and len(repaired_validation.rejected) < len(primary_validation.rejected)
            ):
                validation = repaired_validation
            else:
                validation = primary_validation

        escalate_reason = None
        if not validation.usable and entities:
            escalate_reason = "schema_empty_or_all_rejected"
        elif validation.rejected:
            # Partial but usable: keep accepted edges, drop rejects with audit.
            # Do not tooler-escalate — that was discarding happy-path extracts.
            print(
                "[DEBUG (run_batch_extract)]: dropping rejected items "
                f"count={len(validation.rejected)} reasons={validation.reasons}"
            )
        if (
            validation.usable
            and event_leg_incomplete(validation.accepted, entities)
        ):
            print(
                "[DEBUG (run_batch_extract)]: event_leg_incomplete after "
                "schema extract; persisting accepted without escalate"
            )

        accepted_rels: List[ArchitectAgentRelationship] = []
        if validation.accepted:
            for rel in validation.accepted:
                props = dict(rel.properties or {})
                if rel.source_span and not props.get("source_span"):
                    props["source_span"] = rel.source_span
                if rel.span_start is not None:
                    props["span_start"] = rel.span_start
                if rel.span_end is not None:
                    props["span_end"] = rel.span_end
                if rel.happened_at and not props.get("happened_at"):
                    props["happened_at"] = rel.happened_at
                accepted_rels.append(
                    _to_architect_relationship(
                        {
                            "tail": rel.tail.model_dump(mode="json"),
                            "tip": rel.tip.model_dump(mode="json"),
                            "name": rel.name,
                            "description": rel.description,
                            "amount": rel.amount,
                            "properties": props,
                        }
                    )
                )
            if accepted_rels and escalate_reason is None:
                self.queue_relationships_for_persistence(accepted_rels)

        should_escalate = bool(escalate and escalate_reason)
        if cost_ledger is not None:
            cost_ledger.record_architect_unit(
                escalated=should_escalate,
                reason=escalate_reason,
                schema_calls=schema_calls,
                repair_calls=repair_calls,
            )

        if should_escalate:
            max_turns = max(0, int(config.ingest_architect_escalate_max_turns))
            if max_turns <= 0:
                print(
                    "[DEBUG (run_batch_extract)]: escalate suppressed "
                    f"reason={escalate_reason} (max_turns=0); partial/empty unit"
                )
                if cost_ledger is not None:
                    cost_ledger.janitor_drop_reasons.append(
                        "escalate_budget_exhausted"
                    )
                return accepted_rels
            print(
                f"[DEBUG (run_batch_extract)]: escalating unit reason={escalate_reason} "
                f"max_turns={max_turns}"
            )
            return self.run_tooler(
                text,
                entities,
                targeting=targeting,
                brain_id=brain_id,
                timeout=timeout,
                max_retries=max_retries,
                ingestion_session_id=ingestion_session_id,
                mode=mode,
                reset=False,
                max_tool_turns=max_turns,
            )

        return list(self.relationships_set) if self.relationships_set else accepted_rels

    @routing_stage("architect")
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
        reset: bool = True,
        max_tool_turns: Optional[int] = None,
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
            reset (bool): When True, clear prior session relationships before running.
            max_tool_turns (Optional[int]): Hard cap on custom-backend tool loop turns (escalate budget).

        Returns:
            List[ArchitectAgentRelationship]: The relationships discovered and collected by the agent during this run.

        Raises:
            TimeoutError: If the agent fails to produce a response within `timeout` after the allowed retry attempts.
        """
        import uuid

        from src.lib.redis.client import _redis_client

        if reset or not self.session_id:
            self.session_id = str(uuid.uuid4())
            self.relationships_set.clear()
            self.pending_persistence_batches.clear()

        entities_dict = {
            entity.uuid: strip_properties([entity.model_dump(mode="json")])[0]
            for entity in entities
        }
        if reset:
            self.entities = entities_dict
        else:
            existing = getattr(self, "entities", None) or {}
            existing.update(entities_dict)
            self.entities = existing

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
            invoke_config = {
                "recursion_limit": MAX_RECURSION_LIMIT,
                "tags": ["architect_agent", "architect_tooler"],
                "metadata": metadata,
            }
            if max_tool_turns is not None:
                invoke_config["max_tool_turns"] = int(max_tool_turns)
            return self.agent.invoke(
                {"messages": messages_list},
                config=invoke_config,
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
                    future = submit_with_context(executor, _invoke_agent, previous_messages)
                    response = future.result(timeout=timeout)
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
