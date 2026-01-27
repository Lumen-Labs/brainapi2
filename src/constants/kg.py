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
from typing import List, Literal, Optional, Tuple
import uuid
from pydantic import BaseModel, ConfigDict, Extra, Field

from src.constants.data import Observation


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

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the node was last updated.",
    )

    observations: Optional[List[Observation]] = Field(
        default=None, description="The observations of the node."
    )

    class Meta:
        """
        Allow extra properties to be added to the node.
        """

        extra = Extra.allow


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
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the predicate was last updated.",
    )
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
    connected_by: Tuple[Node, Node]
