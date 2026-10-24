"""
File: /requests.py
Created Date: Monday October 20th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday April 12th 2026 1:35:36 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from src.constants.data import Observation, PartialNode, PartialPredicate, TextChunk
from src.constants.kg import (
    EntitySynergy,
    IdentificationParams,
    Node,
    Predicate,
    Relationship,
)
from src.constants.tasks.ingestion import IngestionTaskArgs
from src.core.search.entity_info import MatchPath


class IngestionRequestBody(IngestionTaskArgs):
    """
    Request body for the ingestion endpoint.
    """


class IngestionUrlRequestBody(BaseModel):
    url: str = Field(description="Public https URL to crawl.")
    max_pages: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum pages to crawl. Each cleaned page becomes one ingest job.",
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="Extra eTLD+1 hosts to include, max 20.",
    )
    skip_enrichment: bool = Field(
        default=False,
        description="Persist cleaned page text and embeddings only. Skip Scout/Architect.",
    )

    @field_validator("url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        parsed = urlparse((value or "").strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("url must be a public https URL")
        return (value or "").strip()

    @field_validator("allowed_domains")
    @classmethod
    def validate_allowed_domains(cls, value: List[str]) -> List[str]:
        if value is None:
            return []
        if len(value) > 20:
            raise ValueError("allowed_domains accepts at most 20 hosts")
        return [item.strip() for item in value if item and str(item).strip()]


class IngestionStructuredDataElement(BaseModel):
    """
    Element for the structured ingestion endpoint.
    """

    json_data: dict = Field(
        default={},
        description="The data rapresenting the structured element.",
    )
    metadata: Optional[dict] = Field(
        default={},
        description="The metadata of the structured element. The information here will be appended to the entity but not analyzed.",
    )
    types: List[str] = Field(
        default=[],
        description="A list of types, used to categorize the data.",
    )
    identification_params: Optional[IdentificationParams] = Field(
        default=None,
        description="The parameters used to identify the structured element.",
    )
    textual_data: Optional[dict] = Field(
        default={},
        description="The textual descriptive data rapresenting the structured element. Could be a description, a summary, some notes, etc.",
    )


class RequestPartialNode(PartialNode):
    uuid: Optional[str] = Field(None, description="The id of the node.")
    type: str = Field(description="The type of the node.")
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional labels; defaults to [type] when omitted.",
    )
    description: Optional[str] = Field(
        default=None,
        description="The description of the node.",
    )
    properties: Optional[dict] = Field(
        default_factory=dict,
        description="The properties of the node.",
    )
    happened_at: Optional[str] = Field(
        None,
        description="The date and time the node happened at if known otherwise None. Mostly used for event nodes.",
    )

    @model_validator(mode="after")
    def default_labels_from_type(self):
        if not self.labels:
            self.labels = [self.type]
        if self.properties is None:
            self.properties = {}
        return self


class RequestPartialPredicate(PartialPredicate):
    uuid: Optional[str] = Field(None, description="The id of the relationship.")
    description: Optional[str] = Field(
        default=None,
        description="The description of the relationship.",
    )
    amount: Optional[Union[int, float]] = Field(
        None,
        description="The amount of the relationship, if it is a quantitative relationship.",
    )


class IngestionTripleSet(BaseModel):
    subject: Optional[RequestPartialNode] = Field(
        None,
        description="The optional actor of the triple.",
    )
    subj_event: Optional[RequestPartialPredicate] = Field(
        None,
        description="The optional event predicate of the triple.",
    )
    event: Optional[RequestPartialNode] = Field(
        None,
        description="The optional event node. Omit for a direct subject-predicate-object edge.",
    )
    event_obj: Optional[RequestPartialPredicate] = Field(
        None,
        description="The optional event-to-object predicate. Paired with event when present.",
    )
    object: RequestPartialNode

    @model_validator(mode="after")
    def require_event_pair_or_direct_edge(self):
        has_event = self.event is not None
        has_event_obj = self.event_obj is not None
        if has_event != has_event_obj:
            raise ValueError("event and event_obj must both be set or both omitted")
        if has_event:
            return self
        if not self.subject or not self.subj_event:
            raise ValueError(
                "direct triples require subject, subj_event, and object"
            )
        return self


class PartialNodeFilter(BaseModel):
    """
    Partial node filter model.
    """

    name: Optional[str] = Field(
        None,
        description="The name of the node to filter by.",
    )
    type: Optional[str] = Field(
        None,
        description="The type of the node to filter by.",
    )
    uuid: Optional[str] = Field(
        None,
        description="The uuid of the node to filter by.",
    )
    meta_description: Optional[str] = Field(
        None,
        description="An extra description that explains what this node is about, helpful for entity resolution.",
    )

    @model_validator(mode="after")
    def require_uuid_or_name_and_type(self):
        if self.uuid:
            return self
        if self.name and self.type:
            return self
        raise ValueError(
            "Anchor must include either uuid, or both name and type."
        )


class IngestionStructuredRequestBody(BaseModel):
    """
    Request body for the structured ingestion endpoint.
    """

    data: List[IngestionTripleSet] = Field(
        ...,
        description="The list of event-centric information triples to ingest.",
    )
    anchor: Optional[PartialNodeFilter] = Field(
        None,
        description="The related information to connect the structured data to.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Additional text context for the structured data.",
    )
    mode: Optional[Literal["deterministic", "hybrid", "enrich"]] = Field(
        default=None,
        description=(
            "Ingest mode. deterministic: no LLM (triples only; UUID anchor or "
            "exact name+type match). hybrid: persist submitted triples then "
            "optionally enrich from text. enrich: same as hybrid with LLM "
            "backfill. When omitted, inferred as hybrid if text is set else "
            "deterministic."
        ),
    )
    brain_id: str = Field(
        default="default", description="The brain identifier to store the data in."
    )

    @field_validator("anchor", mode="before")
    @classmethod
    def reject_string_anchor(cls, value):
        if isinstance(value, str):
            raise ValueError(
                "Anchor must be an object with uuid or name+type, not a string."
            )
        return value

    def resolved_mode(self) -> Literal["deterministic", "hybrid", "enrich"]:
        if self.mode is not None:
            return self.mode
        return "hybrid" if self.text else "deterministic"


class RetrieveRequestResponse(BaseModel):
    """
    Response for the retrieve endpoint.
    """

    data: List[TextChunk]
    observations: List[Observation]
    relationships: List[dict]

    @field_serializer("relationships", when_used="json")
    def _serialize_relationships(self, value: List[Any]):
        try:
            from neo4j.graph import (
                Node as NeoNode,
            )
            from neo4j.graph import (
                Path as NeoPath,
            )
            from neo4j.graph import (
                Relationship as NeoRel,
            )
        except Exception:
            NeoNode = NeoRel = NeoPath = tuple()
        from src.constants.kg import Node as KGNode

        def _ser(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, KGNode):
                return obj.model_dump(mode="json")
            if isinstance(obj, dict):
                return {k: _ser(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_ser(v) for v in obj]
            if NeoNode and isinstance(obj, NeoNode):
                d = dict(obj)
                labels = getattr(obj, "labels", None)
                if labels is not None:
                    d["_labels"] = list(labels)
                return d
            if NeoRel and isinstance(obj, NeoRel):
                d = dict(obj)
                rel_type = getattr(obj, "type", None)
                if rel_type is not None:
                    d["_type"] = rel_type
                try:
                    nodes_attr = getattr(obj, "nodes", None)
                    if nodes_attr:
                        d["_start_id"] = getattr(nodes_attr[0], "id", None)
                        d["_end_id"] = getattr(nodes_attr[1], "id", None)
                except Exception:
                    pass
                return d
            if NeoPath and isinstance(obj, NeoPath):
                return {
                    "nodes": [_ser(n) for n in obj.nodes],
                    "relationships": [_ser(r) for r in obj.relationships],
                }
            return str(obj)

        return [_ser(v) for v in value] if value is not None else []


class RetrievedNeighborNode(BaseModel):
    """
    Node model for the retrieve neighbors endpoint.
    """

    neighbor: Node
    relationship: Predicate
    most_common: Node
    similarity_score: float = Field(
        0.0, description="The similarity score of the neighbor to the main node."
    )


class RetrieveNeighborsRequestResponse(BaseModel):
    """
    Response for the retrieve neighbors endpoint.
    """

    count: int = Field(0, description="The number of neighbors found.")
    main_node: Node = Field(..., description="The main node of the neighbors.")
    neighbors: List[RetrievedNeighborNode]


class RetrieveNeighborsAiModeRequestBody(BaseModel):
    """
    Request body for the retrieve neighbors AI mode endpoint.
    """

    identification_params: IdentificationParams = Field(
        ...,
        description="The identification parameters of the entity to get neighbors for.",
    )
    limit: int = Field(10, description="The number of neighbors to return.")
    looking_for: Optional[list[str]] = Field(
        ...,
        description="The description of the neighbors to look for.",
    )
    brain_id: str = Field(
        default="default", description="The brain identifier to store the data in."
    )


class RetrieveNeighborsWithIdentificationParamsRequestBody(BaseModel):
    """
    Request body for the retrieve neighbors with identification params endpoint.
    """

    identification_params: IdentificationParams = Field(
        ...,
        description="The identification parameters of the entity to get neighbors for.",
    )
    limit: int = Field(10, description="The number of neighbors to return.")
    brain_id: str = Field(
        default="default", description="The brain identifier to store the data in."
    )
    look_for: Optional[str] = Field(
        None, description="Optional filter for what type of neighbors to look for."
    )


class CreateBrainRequest(BaseModel):
    """
    Request body for the create brain endpoint.
    """

    brain_id: str


class AddEntityRequest(BaseModel):
    """Request model for adding a new entity to the graph."""

    name: str
    brain_id: str = "default"
    labels: list[str] = []
    description: Optional[str] = None
    properties: Optional[dict] = None
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None


class UpdateEntityRequest(BaseModel):
    """Request model for updating an existing entity in the graph."""

    uuid: str
    brain_id: str = "default"
    new_name: Optional[str] = None
    new_description: Optional[str] = None
    new_labels: Optional[list[str]] = None
    new_properties: Optional[dict] = None
    properties_to_remove: Optional[list[str]] = None


class AddRelationshipRequest(BaseModel):
    """Request model for adding a new relationship between two entities."""

    subject_uuid: str
    predicate_name: str
    predicate_description: str
    object_uuid: str
    brain_id: str = "default"


class UpdateRelationshipRequest(BaseModel):
    """Request model for updating an existing relationship's properties."""

    uuid: str
    brain_id: str = "default"
    new_properties: Optional[dict] = None
    properties_to_remove: Optional[list[str]] = None


class GetEntityInfoResponse(BaseModel):
    """Response model for the get entity info endpoint."""

    target_node: Optional[Node] = None
    path: MatchPath


class GetEntityContextResponse(BaseModel):
    """Response model for the get entity context endpoint."""

    neighborhood: list[dict]
    target_node: Optional[Node] = None
    text_contexts: list[str] = []
    natural_language_web: list[dict] = []


class GetEntitySibilingsResponse(BaseModel):
    """Response model for the get entity siblings endpoint."""

    target_node: Node
    synergies: List[EntitySynergy]
    anchors: Optional[List[Node]] = None
    potential_anchors: Optional[List[Node]] = None


class RecommendRequestBody(BaseModel):
    """Request body for event-graph recommendations."""

    target: str
    brain_id: str = "default"
    polarity: Literal["same", "opposite"] = "same"
    top_k: int = Field(20, ge=1, le=200)
    labels: Optional[List[str]] = None
    include_asymmetric: bool = True
    include_multi_interest: bool = True
    include_attribute_pref: bool = False
    diversify: bool = True
    asymmetric_direction: Literal["outbound", "inbound", "both"] = "outbound"
    exclude_seen: bool = False
    recency_half_life_days: Optional[float] = Field(None, ge=0)
    dampen_degree: bool = False
    behavior_weights: Optional[Dict[str, float]] = None


class RecommendItem(BaseModel):
    node: Node
    score: float
    connected_by: List[Node]
    channel: str


class RecommendResponse(BaseModel):
    target_node: Node
    recommendations: List[RecommendItem]


class GetEntityStatusResponse(BaseModel):
    """Response model for the get entity status endpoint."""

    node: Optional[Node] = None
    exists: bool
    has_relationships: bool
    relationships: List[Tuple[Predicate, Node]]
    observations: List[Observation]


class GetContextRequestBody(BaseModel):
    """Request body for the get context endpoint."""

    text: str
    brain_id: str = "default"
    historical_limit: int = 10
    max_facts: int = Field(40, ge=0)
    max_passages: int = 8
    apply_fact_filter: bool = True
    use_ppr: bool = True
    sufficiency_retry: bool = False
    profile_stages: bool = False
    cross_event_bridges: int = Field(3, ge=0)


class GetContextTriple(BaseModel):
    """Triple for the get context endpoint."""

    identified_entity: str
    triple: Tuple[Node, Predicate, Node, Predicate, Node]
    source_chunk_ids: Optional[List[str]] = None
    source_session_ids: Optional[List[str]] = None


class GetContextResponse(BaseModel):
    """Response for the get context endpoint."""

    text_context: str
    triples: List[GetContextTriple]
    historical_context: List[str] = []
    source_passages: List[str] = []
    graph_session_ids: Optional[List[str]] = None
    temporal_conflicts: Optional[List[dict[str, Any]]] = None
    paths: Optional[List[dict[str, Any]]] = None
    topics: Optional[List[dict[str, Any]]] = None
    stage_timings: Optional[dict[str, Any]] = None


class SearchRequestBody(BaseModel):
    query: str
    brain_id: str = "default"
    k: int = Field(10, ge=1, le=200)
    channels: List[str] = Field(default_factory=lambda: ["passages"])
    node_labels: Optional[List[str]] = None
    community_labels: Optional[List[str]] = None
    expand: Literal["none", "neighbors"] = "none"
    fusion: Optional[Literal["rrf", "cc"]] = None
    fusion_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rerank: Optional[str] = Field(
        default=None,
        description="none or plugin:<name>. Unknown plugin names return 400.",
    )
    mode: Literal["default", "catalog"] = "default"
    profile_stages: bool = False
    extras: Optional[Dict[str, str]] = None
    target: Optional[str] = None

    @field_validator("target", mode="before")
    @classmethod
    def empty_target(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("extras", mode="before")
    @classmethod
    def coerce_extras_filter(cls, value):
        if value is None or value == {}:
            return None
        if not isinstance(value, dict):
            raise ValueError("extras must be an object of string keys and values")
        return {
            str(key): str(item)
            for key, item in value.items()
            if item is not None
        } or None


class SearchHitScores(BaseModel):
    bm25: Optional[float] = None
    dense: Optional[float] = None
    rrf: Optional[float] = None
    cc: Optional[float] = None
    rerank: Optional[float] = None
    plugin: Optional[dict[str, float]] = None
    graph: Optional[float] = None
    personalize: Optional[float] = None


class SearchHit(BaseModel):
    id: str
    channel: str
    score: float
    scores: SearchHitScores
    snippet: str
    labels: List[str] = Field(default_factory=list)
    extras: Optional[dict[str, Any]] = None
    node_id: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit]
    stage_timings: Optional[dict[str, Any]] = None
    channel_lists: Optional[dict[str, list[str]]] = None
    facets: Optional[dict[str, dict[str, int]]] = None
    node_ids: List[str] = Field(default_factory=list)
