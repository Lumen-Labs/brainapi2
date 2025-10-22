"""
File: /kg.py
Created Date: Wednesday October 22nd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday October 22nd 2025 8:09:09 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
import uuid
from pydantic import BaseModel, Extra, Field


class Node(BaseModel):
    """
    Node model.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    name: str
    description: Optional[str]
    properties: dict = Field(default_factory=dict)

    class Meta:
        """
        Allow extra properties to be added to the node.
        """

        extra = Extra.allow


class Triple(BaseModel):
    """
    Triple model.
    """

    subject: Node
    predicate: str
    object: Node
