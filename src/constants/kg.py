"""
File: /kg.py
Created Date: Wednesday October 22nd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from datetime import datetime
from typing import List, Literal, Optional, Tuple, TypedDict, Union
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants.data import Observation


def _coerce_datetime(value):
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        try:
            native = to_native()
            if isinstance(native, datetime):
                return native
        except Exception:
            pass
    iso = getattr(value, "isoformat", None)
    if callable(iso) and not isinstance(value, str):
        try:
            value = iso()
        except TypeError:
            return None
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


class Node(BaseModel):
    """
    Node model.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    labels: List[str]
    name: str
    description: Optional[str] = None
    properties: dict = Field(default_factory=dict)

    polarity: Optional[Literal["positive", "negative", "neutral"]] = Field(
        default="neutral",
        description="The polarity of the node.",
    )

    metadata: Optional[dict] = Field(
        default=None, description="The metadata of the node."
    )

    happened_at: Optional[str | None] = Field(
        default=None,
        description="The date and time the node happened at if known otherwise None. Mostly used for event nodes.",
    )

    last_updated: Optional[datetime] = Field(
        default=None,
        description="The date and time the node was last updated.",
    )

    @field_validator("last_updated", mode="before")
    @classmethod
    def _coerce_node_last_updated(cls, value):
        return _coerce_datetime(value)

    observations: Optional[List[Observation]] = Field(
        default=None, description="The observations of the node."
    )

    verification_state: Optional[Literal["unverified", "verified", "rejected"]] = Field(
        default="unverified",
        description="Expert verification state of this fact-representing node.",
    )
    verified_by: Optional[str] = Field(
        default=None, description="User or agent identity that last verified or rejected this node."
    )
    verified_at: Optional[str] = Field(
        default=None, description="ISO timestamp of the last verification state change."
    )
    verification_source: Optional[str] = Field(
        default=None, description="Source, document, or reference id for the verification."
    )
    verification_note: Optional[str] = Field(
        default=None, description="Optional note attached to the last verification action."
    )

    @model_validator(mode="after")
    def _hydrate_verification(self):
        from src.core.verification.facts import hydrate_entity_verification

        return hydrate_entity_verification(self)

    class Meta:
        """
        Allow extra properties to be added to the node.
        """

        extra = "allow"


class Predicate(BaseModel):
    """
    Predicate model.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    flow_key: Optional[str] = Field(
        default=None,
        description="Unique identitier for contextualizing the predicate into the context flow",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="The date and time the predicate was last updated.",
    )

    @field_validator("last_updated", mode="before")
    @classmethod
    def _coerce_predicate_last_updated(cls, value):
        return _coerce_datetime(value)
    deprecated: bool = Field(
        default=False, description="Whether the predicate is deprecated."
    )

    direction: Optional[Literal["in", "out", "neutral"]] = Field(
        default="neutral",
        description="The direction of the predicate.",
    )
    amount: Optional[float] = Field(
        default=None,
        description="The amount of the predicate.",
    )

    observations: Optional[List[Observation]] = Field(
        default=None, description="The observations of the predicate."
    )

    properties: dict = Field(default_factory=dict)

    verification_state: Optional[Literal["unverified", "verified", "rejected"]] = Field(
        default="unverified",
        description="Expert verification state of this fact relationship.",
    )
    verified_by: Optional[str] = Field(
        default=None,
        description="User or agent identity that last verified or rejected this fact.",
    )
    verified_at: Optional[str] = Field(
        default=None, description="ISO timestamp of the last verification state change."
    )
    verification_source: Optional[str] = Field(
        default=None, description="Source, document, or reference id for the verification."
    )
    verification_note: Optional[str] = Field(
        default=None, description="Optional note attached to the last verification action."
    )

    @model_validator(mode="after")
    def _hydrate_verification(self):
        from src.core.verification.facts import hydrate_entity_verification

        return hydrate_entity_verification(self)

    level: Optional[Literal["1", "2", "3"]] = Field(
        default=None,
        description=(
            "The level of the predicate. "
            "1: directly connected to the main node. "
            "2: connected to similar nodes of same label, not directly connected."
            "3: connected to similar nodes of different label, not directly connected."
        ),
    )


class Triple(BaseModel):
    """
    Triple model.
    """

    subject: Node
    predicate: Predicate
    object: Node


class Relationship(BaseModel):
    """
    Relationship model.
    """

    direction: Literal["in", "out", "neutral"] = Field(
        default="neutral",
        description=(
            "The direction of the relationship. always referred to the left/first node.",
            "eg: Tuple[node, predicate, node] -> direction 'in' means inbound to the first node"
            "so going from the last node to the first node.",
        ),
    )
    predicate: Predicate
    subject: Node
    object: Node


class IdentificationParams(BaseModel):
    """
    Identification params model.
    Must have a name property.
    Than can contain other properties.
    """

    name: str
    entity_types: List[str] = Field(
        default=[],
        description="The types of the entity to identify.",
    )

    model_config = ConfigDict(extra="allow")


class SearchRelationshipsResult(BaseModel):
    """
    Search relationships result model.
    """

    results: List[Triple]
    total: int


class SearchEntitiesResult(BaseModel):
    """
    Search entities result model.
    """

    results: List[Node]
    total: int


class EntityInfo(BaseModel):
    """
    Entity info model.
    """

    most_relevant_node: Node
    relevant_nodes: List[Node]


class EntitySynergy(BaseModel):
    """
    Entity synergy model.
    """

    node: Node
    connected_by: List[Node]
    association_score: float


class NodeDict(TypedDict, total=False):
    uuid: str
    name: str
    labels: list[str]


class PredicateDict(TypedDict, total=False):
    uuid: str
    name: str
