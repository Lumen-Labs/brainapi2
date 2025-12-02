"""
File: /data.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 11:48:51 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from datetime import datetime
from typing import List, Optional
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


class KGChanges(BaseModel):
    """
    KG changes model.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    node_ids: Optional[List[str]] = Field(
        default=None, description="The id of the node that was changed."
    )
    predicate_id: Optional[str] = Field(
        default=None, description="The id of the predicate that was changed."
    )

    changelog: List[dict] = Field(description="The changelog of the changes.")
    date: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the changes were made.",
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
