"""
File: /requests.py
Created Date: Monday October 20th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday October 20th 2025 7:29:39 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from turtle import st
from typing import List, Any, Optional
from pydantic import BaseModel, Field, field_serializer
from src.constants.data import Observation, TextChunk
from src.constants.kg import IdentificationParams, Node, Relationship
from src.constants.tasks.ingestion import IngestionTaskArgs


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
    store: Optional[str] = Field(
        default="default",
        description="The store where the data will be stored, if not provided, the data will be stored in the default store.",
    )
    observate_for: Optional[List[str]] = Field(
        default=[],
        description="What to look for and describe in the data during observation. "
        "If not provided, the observations will be generic",
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


class RetrievedNeighborNode(Node):
    """
    Node model for the retrieve neighbors endpoint.
    """

    relation: Relationship
    observations: List[str]


class RetrieveNeighborsRequestResponse(BaseModel):
    """
    Response for the retrieve neighbors endpoint.
    """

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


class RetrieveNeighborsWithIdentificationParamsRequestBody(BaseModel):
    """
    Request body for the retrieve neighbors with identification params endpoint.
    """

    identification_params: IdentificationParams = Field(
        ...,
        description="The identification parameters of the entity to get neighbors for.",
    )
    limit: int = Field(10, description="The number of neighbors to return.")
