"""Typed response envelopes for the public OpenAPI contract."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, JsonValue, RootModel

from src.constants.data import KGChanges, Observation, StructuredData, TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate, Triple


class IngestionAcceptedResponse(BaseModel):
    message: str
    task_id: str


class LoginInfoResponse(BaseModel):
    is_system_pat: bool
    brain_id: str


class BrainLifecycleResponse(BaseModel):
    success: bool
    brain_id: str
    operation: Literal["clear", "delete"]
    existed: bool
    cleared_backends: list[str]


class StringListResponse(RootModel[list[str]]):
    pass


class FlatHopNode(BaseModel):
    uuid: str
    labels: list[str]
    name: str


class FlatHopPredicate(BaseModel):
    uuid: str
    name: str
    direction: str | None = None


HopNode = Node | FlatHopNode
HopPredicate = Predicate | FlatHopPredicate
SecondDegreeHop = tuple[
    HopNode,
    list[tuple[HopPredicate, HopNode, list[tuple[HopPredicate, HopNode]]]],
]


class HopListResponse(RootModel[list[SecondDegreeHop]]):
    pass


class RelationshipListResponse(BaseModel):
    message: str
    relationships: list[Triple]
    total: int


class EntityListResponse(BaseModel):
    message: str
    entities: list[Node]
    total: int


class StructuredDataItemResponse(BaseModel):
    message: str
    data: StructuredData


class StructuredDataListResponse(BaseModel):
    message: str
    data: list[StructuredData]
    count: int
    total: int


class TypeListResponse(BaseModel):
    message: str
    types: list[str]
    count: int


class TextChunkListResponse(BaseModel):
    message: str
    data: list[TextChunk]
    total: int


class ObservationItemResponse(BaseModel):
    message: str
    observation: Observation


class ObservationListResponse(BaseModel):
    message: str
    observations: list[Observation]
    count: int


class LabelListResponse(BaseModel):
    message: str
    labels: list[str]
    count: int


class ChangelogItemResponse(BaseModel):
    message: str
    changelog: KGChanges


class ChangelogListResponse(BaseModel):
    message: str
    changelogs: list[KGChanges]
    count: int


class VectorStoreInfo(BaseModel):
    name: str
    dimension: int


class VectorStoreListResponse(BaseModel):
    stores: list[VectorStoreInfo]


class VectorListResponse(BaseModel):
    message: str
    store: str
    vectors: list[Vector]
    total: int


class TaskStateResponse(BaseModel):
    """Task payload with stable state fields and pipeline-specific extensions."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    id: str | None = None
    status: str
    stage: str | None = None
    error: str | None = None
    result: JsonValue | None = None


class TaskListResponse(BaseModel):
    tasks: list[TaskStateResponse]


class IngestGroupStageTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    calls: int = 0



class IngestGroupMeterResponse(BaseModel):
    group_tracking: str
    task_count: int
    completed: int
    failed: int
    partial_failed: int
    total_llm_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int = 0
    source_chars: int
    source_tokens: int
    stages: dict[str, IngestGroupStageTotals]
    updated_at: Optional[str] = None
    task_ids: list[str] = []


class IngestionCapabilitiesResponse(BaseModel):
    progressive: bool
    version: str
    concurrency: int


class IngestionActivityResponse(RootModel[dict[str, JsonValue]]):
    pass
