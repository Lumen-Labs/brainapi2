from src.core.saving.pipeline_runtime import routing_stage
"""
File: /scout_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday March 22nd 2026 10:11:53 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from src.core.saving.ingest_cost import submit_with_context
from typing import Dict, List, Literal, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
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
from src.constants.kg import Node
from src.constants.prompts.scout_agent import (
    SCOUT_AGENT_COARSE_EXTRACT_ENTITIES_PROMPT,
    SCOUT_AGENT_COARSE_SYSTEM_PROMPT,
    SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT,
    SCOUT_AGENT_EXTRACT_STRUCTURED_ENTITIES_PROMPT,
    SCOUT_AGENT_SYSTEM_PROMPT,
)
from src.core.agents.core import parse_structured_from_messages, runtime_agent_factory
from src.core.agents.core.parsing import get_finish_reason_from_response
from src.core.plugins.prompts import prompt_registry
from src.services.api.constants.requests import IngestionTripleSet
from src.utils.text_chunking import chunk_text as _chunk_text


class _ScoutEntity(BaseModel):
    """
    Scout entity.
    """

    type: str
    name: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None
    happened_at: Optional[str] = Field(
        default=None,
        description="The date and time the entity happened at if known otherwise None.",
    )
    polarity: Optional[Literal["positive", "negative", "neutral"]] = Field(
        default="neutral",
        description="The polarity of the entity.",
    )


class ScoutEntity(_ScoutEntity):
    """
    Scout entity.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class _ScoutAgentResponse(BaseModel):
    """
    Scout agent response containing the extracted entities.
    """

    entities: List[_ScoutEntity]


class ScoutAgentResponse(BaseModel):
    """
    Scout agent response containing the extracted entities (subjects and objects)
    from the text with their properties.
    """

    entities: List[ScoutEntity]


class ScoutAgent:
    """
    Scout agent.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache_adapter: CacheAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
    ):
        """
        Initialize a ScoutAgent with the provided adapters and reset internal agent state.

        Stores the provided LLM, cache, knowledge graph, vector store, and embeddings adapters on the instance and sets the agent to None.
        """
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None

    def _get_tools(self, brain_id: str = "default") -> List[BaseTool]:
        """
        Provide the list of tools available to the agent for a given brain identifier.

        Parameters:
            brain_id (str): Identifier of the brain/context to retrieve tools for; defaults to "default".

        Returns:
            List[BaseTool]: A list of BaseTool instances available to the agent for the specified brain.
        """
        return []

    def _get_agent(
        self,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
        brain_id: str = "default",
        mode: Literal["granular", "coarse"] = "granular",
    ):
        if mode == "granular":
            system_prompt = prompt_registry.get(
                "SCOUT_AGENT_SYSTEM_PROMPT", SCOUT_AGENT_SYSTEM_PROMPT
            ).format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        elif mode == "coarse":
            system_prompt = prompt_registry.get(
                "SCOUT_AGENT_COARSE_SYSTEM_PROMPT", SCOUT_AGENT_COARSE_SYSTEM_PROMPT
            ).format(
                extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
            )
        else:
            raise ValueError(f"Invalid mode for scout agent: {mode}")

        self.agent = runtime_agent_factory.build(
            model=self.llm_adapter.llm.langchain_model,
            tools=(tools if tools else self._get_tools(brain_id)),
            system_prompt=system_prompt,
            output_schema=output_schema if output_schema else None,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            architecture=config.agentic_architecture,
            use_custom_backend=(mode == "coarse"),
        )

    @routing_stage("scout")
    def run_structured(
        self,
        text: str,
        brain_id: str = "default",
        timeout: int = 300,
        max_retries: int = 3,
        ingestion_session_id: Optional[str] = None,
        partial_triples: List[IngestionTripleSet] = [],
        current_triples: List[IngestionTripleSet] = [],
    ) -> ScoutAgentResponse:
        """
        Given the provided text and partial/full triples, extract all the (missing) entities from the text.

        Performs an LLM invocation (with optional targeting context), applies retries with exponential backoff on timeouts, and enforces a per-invocation timeout.

        Parameters:
            text: The input text to extract entities from.
            targeting: Optional Node providing contextual targeting information (name, description, properties) to bias extraction.
            brain_id: Identifier for the agent/brain configuration to use.
            timeout: Maximum seconds to wait for a single agent invocation before treating it as a timeout.
            max_retries: Maximum number of retry attempts for timed-out invocations using exponential backoff.
            ingestion_session_id: Identifier for the ingestion session to use.
            mode: Mode to use for the scout agent. "granular" for a more granular extraction, "coarse" to extract the most important entities only.
            partial_triples: List of partial triples to use for the extraction.
            current_triples: List of current triples to use for the extraction.
        Returns:
            A ScoutAgentResponse containing:
                - entities: list of extracted ScoutEntity objects.

        Raises:
            TimeoutError: If a single invocation exceeds `timeout`, or if all retry attempts fail due to timeouts.
        """
        self._get_agent(
            output_schema=_ScoutAgentResponse,
            brain_id=brain_id,
            mode="granular",
        )
        current_entities = []
        for triple in current_triples:
            if not triple.subject or not triple.object or not triple.event:
                continue
            current_entities.extend(
                [
                    ScoutEntity(
                        type=triple.subject.type,
                        name=triple.subject.name,
                        description=triple.subject.description,
                        properties=triple.subject.properties or {},
                        happened_at=triple.subject.happened_at,
                    ),
                    ScoutEntity(
                        type=triple.object.type,
                        name=triple.object.name,
                        description=triple.object.description,
                        properties=triple.object.properties or {},
                        happened_at=triple.object.happened_at,
                    ),
                    ScoutEntity(
                        type=triple.event.type,
                        name=triple.event.name,
                        description=triple.event.description,
                        properties=triple.event.properties or {},
                        happened_at=triple.event.happened_at,
                    ),
                ]
            )
        for triple in partial_triples:
            current_entities.extend(
                [
                    ScoutEntity(
                        type=triple.object.type,
                        name=triple.object.name,
                        description=triple.object.description,
                        properties=triple.object.properties or {},
                    ),
                    ScoutEntity(
                        type=triple.event.type,
                        name=triple.event.name,
                        description=triple.event.description,
                    ),
                ]
            )
        prompt = prompt_registry.get(
            "SCOUT_AGENT_EXTRACT_STRUCTURED_ENTITIES_PROMPT",
            SCOUT_AGENT_EXTRACT_STRUCTURED_ENTITIES_PROMPT,
        ).format(
            text=text,
            current_entities=current_entities,
        )

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
                config={
                    "tags": ["scout_agent"],
                    "metadata": {
                        "agent": "scout_agent",
                        "brain_id": brain_id,
                        **(
                            {"ingestion_session_id": ingestion_session_id}
                            if ingestion_session_id
                            else {}
                        ),
                    },
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry():
            """
            Invoke the agent in a separate thread and enforce the configured timeout.

            Returns:
                dict: The agent response dictionary.

            Raises:
                TimeoutError: If the agent invocation exceeds the specified timeout.
            """
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = submit_with_context(executor, _invoke_agent)
                    response = future.result(timeout=timeout)
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Scout agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Scout agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        structured = response.get("structured_response")
        if isinstance(structured, dict):
            try:
                structured = _ScoutAgentResponse.model_validate(structured)
            except Exception:
                structured = None
        if structured is None or not getattr(structured, "entities", None):
            fallback = parse_structured_from_messages(
                response.get("messages", []), _ScoutAgentResponse
            )
            if fallback is not None and getattr(fallback, "entities", None):
                structured = fallback
        if structured is None:
            structured = _ScoutAgentResponse(entities=[])
        return ScoutAgentResponse(
            entities=[
                ScoutEntity(**entity.model_dump(mode="json"))
                for entity in structured.entities
            ],
        )

    def run(
        self,
        text: str,
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 300,
        max_retries: int = 3,
        ingestion_session_id: Optional[str] = None,
        mode: Literal["granular", "coarse"] = "granular",
        reference_time: Optional[str] = None,
        preferred_extraction_entities: Optional[List[str]] = None,
        max_chars_per_chunk: int = 6000,
    ) -> ScoutAgentResponse:
        chunks = _chunk_text(text, max_chars=max_chars_per_chunk)
        if len(chunks) == 1:
            return self._run_chunk(
                chunks[0],
                targeting=targeting,
                brain_id=brain_id,
                timeout=timeout,
                max_retries=max_retries,
                ingestion_session_id=ingestion_session_id,
                mode=mode,
                reference_time=reference_time,
                preferred_extraction_entities=preferred_extraction_entities,
            )
        merged: dict[str, ScoutEntity] = {}
        for chunk in chunks:
            response = self._run_chunk(
                chunk,
                targeting=targeting,
                brain_id=brain_id,
                timeout=timeout,
                max_retries=max_retries,
                ingestion_session_id=ingestion_session_id,
                mode=mode,
                reference_time=reference_time,
                preferred_extraction_entities=preferred_extraction_entities,
            )
            for entity in response.entities:
                key = (
                    (entity.name or "").strip().lower(),
                    (entity.type or "").strip().lower(),
                    getattr(entity, "happened_at", None) or "",
                )
                if key not in merged:
                    merged[key] = entity
        return ScoutAgentResponse(entities=list(merged.values()))

    @routing_stage("scout")
    def _run_chunk(
        self,
        text: str,
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 300,
        max_retries: int = 3,
        ingestion_session_id: Optional[str] = None,
        mode: Literal["granular", "coarse"] = "granular",
        reference_time: Optional[str] = None,
        preferred_extraction_entities: Optional[List[str]] = None,
    ) -> ScoutAgentResponse:
        """
        Extract entities from the provided text using the Scout agent and return a structured response containing the entities.
        """
        self._get_agent(
            output_schema=_ScoutAgentResponse,
            brain_id=brain_id,
            mode=mode,
        )

        targeting_str = (
            f"""
                                The information is related to:
                                "{targeting.name}": {targeting.description}
                                {targeting.properties}
                                """
            if targeting
            else ""
        )
        reference_time_str = (
            f"Reference date for resolving relative dates: {reference_time}"
            if reference_time
            else ""
        )
        preferred_str = (
            "Prefer extracting these entity types when present: "
            + ", ".join(preferred_extraction_entities)
            if preferred_extraction_entities
            else ""
        )
        if mode == "granular":
            prompt = prompt_registry.get(
                "SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT",
                SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT,
            ).format(
                text=text,
                targeting=targeting_str,
                reference_time=reference_time_str,
                preferred_entities=preferred_str,
            )
        elif mode == "coarse":
            prompt = prompt_registry.get(
                "SCOUT_AGENT_COARSE_EXTRACT_ENTITIES_PROMPT",
                SCOUT_AGENT_COARSE_EXTRACT_ENTITIES_PROMPT,
            ).format(
                text=text,
                targeting=targeting_str,
                reference_time=reference_time_str,
                preferred_entities=preferred_str,
            )
        else:
            raise ValueError(f"Invalid mode for scout agent: {mode}")

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
                config={
                    "tags": ["scout_agent"],
                    "metadata": {
                        "agent": "scout_agent",
                        "brain_id": brain_id,
                        **(
                            {"ingestion_session_id": ingestion_session_id}
                            if ingestion_session_id
                            else {}
                        ),
                    },
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
                    future = submit_with_context(executor, _invoke_agent)
                    response = future.result(timeout=timeout)
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Scout agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Scout agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        messages = response.get("messages", [])
        finish_reason = (
            get_finish_reason_from_response(messages[-1]) if messages else None
        )
        if str(finish_reason).upper() in {"MAX_TOKENS", "LENGTH", "MAX_LENGTH"}:
            if len(text.strip()) <= 256:
                raise ValueError(
                    "Entity extraction exceeded the model output limit on a short section."
                )
            return self.run(
                text,
                targeting=targeting,
                brain_id=brain_id,
                timeout=timeout,
                max_retries=max_retries,
                ingestion_session_id=ingestion_session_id,
                mode=mode,
                reference_time=reference_time,
                preferred_extraction_entities=preferred_extraction_entities,
                max_chars_per_chunk=len(text.strip()) // 2,
            )

        structured = response.get("structured_response")
        if isinstance(structured, dict):
            try:
                structured = _ScoutAgentResponse.model_validate(structured)
            except Exception:
                structured = None
        if structured is None or not getattr(structured, "entities", None):
            fallback = parse_structured_from_messages(
                response.get("messages", []), _ScoutAgentResponse
            )
            if fallback is not None and getattr(fallback, "entities", None):
                structured = fallback
        if structured is None:
            structured = _ScoutAgentResponse(entities=[])
        return ScoutAgentResponse(
            entities=[
                ScoutEntity(**entity.model_dump(mode="json"))
                for entity in structured.entities
            ],
        )
