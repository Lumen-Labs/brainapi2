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
from typing import Optional
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
