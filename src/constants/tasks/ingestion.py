"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:21:59 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from enum import Enum
from typing import Any, List, Optional, Union, Annotated, Literal
from pydantic import BaseModel, Field, Discriminator, field_validator

from src.core.saving.ingest_groups import InvalidGroupTracking, normalize_group_tracking


class IngestionTaskDataType(Enum):
    """
    Data type for the ingestion task.
    """

    TEXT = "text"
    JSON = "json"


class IngestionTaskJsonArgs(BaseModel):
    """
    Arguments for the ingestion task when the data type is JSON.
    """

    data_type: Literal["json"] = Field(default="json")
    json_data: dict


class IngestionTaskTextArgs(BaseModel):
    """
    Arguments for the ingestion task when the data type is TEXT.
    """

    data_type: Literal["text"] = Field(default="text")
    text_data: str


class IngestionTaskArgs(BaseModel):
    """
    Arguments for the ingestion task.
    """

    data: Annotated[
        Union[IngestionTaskJsonArgs, IngestionTaskTextArgs], Discriminator("data_type")
    ]
    meta_keys: Optional[dict] = Field(
        default=None,
        description=(
            "The shared keys that should not be treated as nodes. "
            "(eg: email because each person has an email). "
            "The keys specified here will not be analyzed "
            "but just added to the metadata."
        ),
    )
    identification_params: Optional[dict] = Field(
        default=None,
        description="The parameters used to identify the data.",
    )
    observate_for: Optional[List[str]] = Field(
        default=[],
        description=(
            "What to look for and describe in the data during observation. "
            "If not provided, the observations will be generic"
        ),
    )
    preferred_extraction_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "The entities that are preferred to be extracted from the data. "
            "If not provided, the extraction will be done on all the entities in the data."
        ),
    )
    brain_id: str = Field(
        default="default", description="The brain identifier to store the data in."
    )
    source_timestamp: Optional[str] = Field(
        default=None,
        description=(
            "Reference timestamp for the source content (e.g. session date). "
            "Used to resolve relative dates like 'yesterday' into absolute dates."
        ),
    )
    skip_enrichment: bool = Field(
        default=False,
        description=(
            "Persist the text chunk and embedding only. Skip observations and "
            "knowledge-graph enrichment (Scout/Architect)."
        ),
    )
    group_tracking: Optional[str] = Field(
        default=None,
        description="Optional ingest-cost rollup key for this request (1-128 chars, [A-Za-z0-9._:-]).",
    )

    @field_validator("group_tracking", mode="before")
    @classmethod
    def _normalize_group_tracking(cls, value: Any) -> Optional[str]:
        try:
            return normalize_group_tracking(value, strict=True)
        except InvalidGroupTracking as exc:
            raise ValueError(str(exc)) from exc
