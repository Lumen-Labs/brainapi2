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
from typing import List, Union, Annotated, Literal
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
    meta_keys: List[str] = Field(
        default=[],
        description=(
            "The shared keys that shuld not be treated as nodes. "
            "(eg: email becasue each person has an email). "
            "The keys specified here will not be analyzed "
            "but just added to the metadata."
        ),
    )


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
