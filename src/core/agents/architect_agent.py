"""
File: /architect_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 2:28:05 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
import uuid
from langchain.agents import create_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
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
from src.constants.kg import Node
from src.constants.prompts.architect_agent import (
    ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT,
    ARCHITECT_AGENT_SYSTEM_PROMPT,
)
from src.core.agents.scout_agent import ScoutEntity

# from src.core.agents.tools.kg_agent import (
#     KGAgentSearchGraphTool,
# )


class ArchitectAgentEntity(BaseModel):
    """
    Architect agent entity.
    """

    uuid: str
    name: str
    type: str


class _ArchitectAgentNew(BaseModel):
    """
    Architect agent new entity.
    """

    temp_id: str
    type: str
    name: str
    reason: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None


class ArchitectAgentNew(BaseModel):
    """
    Architect agent new entity.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    reason: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None


class _ArchitectAgentRelationship(BaseModel):
    """
    Architect agent relationship.
    """

    tip: ArchitectAgentEntity
    name: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None
    tail: ArchitectAgentEntity


class ArchitectAgentRelationship(_ArchitectAgentRelationship):
    """
    Architect agent relationship.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class _ArchitectAgentResponse(BaseModel):
    """
    Architect agent response containing the created relationships
    between the entities.
    """

    new_nodes: List[_ArchitectAgentNew]
    relationships: List[_ArchitectAgentRelationship]


class ArchitectAgentResponse(BaseModel):
    """
    Architect agent response containing the created relationships
    between the entities.
    """

    new_nodes: List[ArchitectAgentNew]
    relationships: List[ArchitectAgentRelationship]
    input_tokens: int
    output_tokens: int


class ArchitectAgent:
    """
    Architect agent.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache_adapter: CacheAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
        # database_desc: str,
    ):
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None
        # self.database_desc = database_desc

    def _get_tools(self, brain_id: str = "default") -> List[BaseTool]:
        return []
        # Currently not using the knowledge graph tools
        # return [
        #     KGAgentSearchGraphTool(
        #         self,
        #         self.kg,
        #         self.vector_store,
        #         self.embeddings,
        #         metadata=metadata,
        #         brain_id=brain_id,
        #     ),
        # ]

    def _get_agent(
        self,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
        brain_id: str = "default",
    ):
        system_prompt = ARCHITECT_AGENT_SYSTEM_PROMPT.format(
            extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
        )

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(tools if tools else self._get_tools(brain_id)),
            system_prompt=system_prompt,
            response_format=output_schema if output_schema else None,
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

        self._get_agent(
            output_schema=_ArchitectAgentResponse,
            brain_id=brain_id,
        )

        self.input_tokens = 0
        self.output_tokens = 0

        def _invoke_agent(
            ent: list[ScoutEntity], all_rels: list[ArchitectAgentRelationship]
        ):
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": ARCHITECT_AGENT_CREATE_RELATIONSHIPS_PROMPT.format(
                                text=text,
                                entities=[
                                    entity.model_dump(mode="json") for entity in ent
                                ],
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
                            ),
                        }
                    ],
                },
            )

        def _process_response(
            response: dict,
            connected_entity_uuids: set,
            all_relationships: list,
            all_new_nodes: list,
            entities: List[ScoutEntity],
        ) -> set:
            structured_response = response.get("structured_response", {})
            iteration_connected = set()

            if hasattr(structured_response, "relationships"):
                for rel in structured_response.relationships:
                    if hasattr(rel, "tip") and hasattr(rel.tip, "uuid"):
                        tip_uuid = rel.tip.uuid
                        if any(e.uuid == tip_uuid for e in entities):
                            iteration_connected.add(tip_uuid)
                    if hasattr(rel, "tail") and hasattr(rel.tail, "uuid"):
                        tail_uuid = rel.tail.uuid
                        if any(e.uuid == tail_uuid for e in entities):
                            iteration_connected.add(tail_uuid)

                if iteration_connected:
                    connected_entity_uuids.update(iteration_connected)
                    all_relationships.extend(structured_response.relationships)

            if hasattr(structured_response, "new_nodes"):
                all_new_nodes.extend(structured_response.new_nodes)

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

            it, ot = _extract_tokens(response)
            self.input_tokens += it
            self.output_tokens += ot

            return iteration_connected

        connected_entity_uuids = set()
        all_relationships = []
        all_new_nodes = []
        max_iterations = 3
        iteration = 0

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry(unconnected_entities_list: List[ScoutEntity]):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _invoke_agent, unconnected_entities_list, all_relationships
                    )
                    response = future.result(timeout=timeout)
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Architect agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        def _invoke_and_process(unconnected_entities_list: List[ScoutEntity]):
            try:
                response = _invoke_agent_with_retry(unconnected_entities_list)
                return _process_response(
                    response,
                    connected_entity_uuids,
                    all_relationships,
                    all_new_nodes,
                    entities,
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

        while len(unconnected_entities) > 0 and iteration <= max_iterations:
            ret = _invoke_and_process(unconnected_entities)
            unconnected_entities = list(
                filter(lambda e: e.uuid not in ret, unconnected_entities)
            )
            print("----------------------------------------------------------")
            from pprint import pprint

            pprint(unconnected_entities)
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
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
