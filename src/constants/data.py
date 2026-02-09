"""
File: /data.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 11:48:51 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union
import uuid
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """
    Text chunk model.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(description="The text of the chunk.")
    metadata: Optional[dict] = None
    inserted_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the chunk was inserted.",
    )
    brain_version: str = Field(
        description="The version of the brain when the chunk was inserted.",
    )


class StructuredData(BaseModel):
    """
    Structured data model.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: dict = Field(description="The json data rapresenting the structured element.")
    types: List[str] = Field(
        description="A list of types, used to categorize the data."
    )
    metadata: Optional[dict] = None
    inserted_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the structured data was inserted.",
    )
    brain_version: str = Field(
        description="The version of the brain when the structured data was inserted.",
        default="0.0.0",
    )


class Observation(BaseModel):
    """
    Observation model.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(description="The text of the observation.")
    metadata: Optional[dict] = None
    resource_id: str = Field(
        description="The id of the resource the observation is about."
    )
    inserted_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the observation was inserted.",
    )


class KGChangesType(Enum):
    """
    KG changes type.
    """

    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_DEPRECATED = "relationship_deprecated"
    NODE_PROPERTIES_UPDATED = "node_properties_updated"
    RELATIONSHIP_PROPERTIES_UPDATED = "relationship_properties_updated"


class PartialNode(BaseModel):
    """
    Partial node model.
    """

    uuid: str = Field(description="The id of the node.")
    name: str = Field(description="The name of the node.")
    labels: List[str] = Field(description="The labels of the node.")
    description: Optional[str] = Field(description="The description of the node.")
    properties: dict = Field(description="The properties of the node.")


class PartialPredicate(BaseModel):
    """
    Partial relationship model.
    """

    uuid: str = Field(description="The id of the relationship.")
    name: str = Field(description="The name of the relationship.")
    description: Optional[str] = Field(
        description="The description of the relationship."
    )
    properties: Optional[dict] = Field(
        default=None, description="The properties of the relationship."
    )


class KGChangeLogRelationshipCreated(BaseModel):
    """
    KG change log relationship created model.
    """

    type: Literal[KGChangesType.RELATIONSHIP_CREATED] = Field(
        default=KGChangesType.RELATIONSHIP_CREATED,
        description="The type of the change.",
    )
    subject: PartialNode = Field(description="The subject of the relationship.")
    predicate: PartialPredicate = Field(
        description="The predicate of the relationship."
    )
    object: PartialNode = Field(description="The object of the relationship.")


class KGChangeLogRelationshipDeprecated(BaseModel):
    """
    KG change log relationship deprecated model.
    """

    type: Literal[KGChangesType.RELATIONSHIP_DEPRECATED] = Field(
        default=KGChangesType.RELATIONSHIP_DEPRECATED,
        description="The type of the change.",
    )
    subject: PartialNode = Field(description="The subject of the relationship.")
    predicate: PartialPredicate = Field(
        description="The predicate of the relationship."
    )
    object: PartialNode = Field(description="The object of the relationship.")
    new_predicate: Optional[PartialPredicate] = Field(
        description="The new predicate of the relationship."
    )


class KGChangeLogPredicateUpdatedProperty(BaseModel):
    """
    KG change log relationship updated property model.
    """

    property: str = Field(description="The property that was updated.")
    previous_value: Any = Field(description="The previous value of the property.")
    new_value: Any = Field(description="The new value of the property.")


class KGChangeLogNodePropertiesUpdated(BaseModel):
    """
    KG change log node properties updated model.
    """

    type: Literal[KGChangesType.NODE_PROPERTIES_UPDATED] = Field(
        default=KGChangesType.NODE_PROPERTIES_UPDATED,
        description="The type of the change.",
    )
    node: PartialNode = Field(description="The node that was updated.")
    properties: List[KGChangeLogPredicateUpdatedProperty] = Field(
        description="The properties that were updated."
    )


class KGChangeLogPredicatePropertiesUpdated(BaseModel):
    """
    KG change log relationship properties updated model.
    """

    type: Literal[KGChangesType.RELATIONSHIP_PROPERTIES_UPDATED] = Field(
        default=KGChangesType.RELATIONSHIP_PROPERTIES_UPDATED,
        description="The type of the change.",
    )
    predicate: PartialPredicate = Field(description="The predicate that was updated.")
    properties: List[KGChangeLogPredicateUpdatedProperty] = Field(
        description="The properties that were updated."
    )


class KGChanges(BaseModel):
    """
    KG changes model.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: KGChangesType = Field(description="The type of the changes.")
    change: Annotated[
        Union[
            KGChangeLogRelationshipCreated,
            KGChangeLogRelationshipDeprecated,
            KGChangeLogNodePropertiesUpdated,
            KGChangeLogPredicatePropertiesUpdated,
        ],
        Field(discriminator="type"),
    ] = Field(description="The change data, discriminated by type.")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp of the changes.",
    )


class Brain(BaseModel):
    """
    Model for a single brain, stored into data db
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name_key: str = Field(description="The key used to identify the brain.")

    @staticmethod
    def _random_pat() -> str:
        import random

        chars = []
        for _ in range(48):
            chars.append(random.choice("abcdefghijklmnopqrstuvwxyz0123456789"))
        return "".join(chars)

    pat: str = Field(
        description="The personal access token for the brain.",
        default_factory=_random_pat,
    )
