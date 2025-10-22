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
from typing import List, Optional, Union, Annotated, Literal
from pydantic import BaseModel, Field, Discriminator


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
