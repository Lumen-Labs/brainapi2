"""
File: /requests.py
Created Date: Monday October 20th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday February 2nd 2026 10:04:06 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_serializer
from src.constants.data import Observation, TextChunk
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


class IngestionStructuredRequestBody(BaseModel):
    """
    Request body for the structured ingestion endpoint.
    """

    data: List[IngestionStructuredDataElement]
    observate_for: Optional[List[str]] = Field(
        default=[],
        description="What to look for and describe in the data during observation. "
        "If not provided, the observations will be generic",
    )
    brain_id: str = Field(
        default="default", description="The brain identifier to store the data in."
    )


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
                Relationship as NeoRel,
                Path as NeoPath,
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


class GetEntityStatusResponse(BaseModel):
    """Response model for the get entity status endpoint."""

    node: Node
    exists: bool
    has_relationships: bool
    relationships: List[Tuple[Predicate, Node]]
    observations: List[Observation]


class GetContextRequestBody(BaseModel):
    """Request body for the get context endpoint."""

    text: str
    brain_id: str = "default"


class GetContextResponse(BaseModel):
    """Response for the get context endpoint."""

    text_context: str
    triples: List[Tuple[Node, Predicate, Node, Predicate, Node]]
