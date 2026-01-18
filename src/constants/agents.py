"""
File: /agents.py
Created Date: Thursday January 15th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 15th 2026 8:55:23 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional, List, Tuple
import uuid


class ArchitectAgentEntity(BaseModel):
    """
    Architect agent entity.
    """

    uuid: str
    name: str
    type: str
    flow_key: Optional[str] = Field(
        default=None,
        description="Unique identitier for contextualizing the entity into the context flow",
    )
    happened_at: Optional[str | None] = Field(
        default=None,
        description="The date and time the entity happened at if known otherwise None. Mostly used for event entities.",
    )


class _ArchitectAgentNew(BaseModel):
    """
    Architect agent new entity.
    """

    temp_id: str
    type: str
    name: str
    reason: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None


class ArchitectAgentNew(BaseModel):
    """
    Architect agent new entity.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    reason: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None


class _ArchitectAgentRelationship(BaseModel):
    """
    Architect agent relationship.
    """

    tip: ArchitectAgentEntity
    name: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None
    tail: ArchitectAgentEntity


class ArchitectAgentRelationship(_ArchitectAgentRelationship):
    """
    Architect agent relationship.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    flow_key: str


class _ArchitectAgentResponse(BaseModel):
    """
    Architect agent response containing the created relationships
    between the entities.
    """

    new_nodes: List[_ArchitectAgentNew]
    relationships: List[_ArchitectAgentRelationship]


class ArchitectAgentResponse(BaseModel):
    """
    Architect agent response containing the created relationships
    between the entities.
    """

    new_nodes: List[ArchitectAgentNew]
    relationships: List[ArchitectAgentRelationship]
    input_tokens: int
    output_tokens: int


class AtomicJanitorAgentWrongRelationship(BaseModel):
    relationship: object
    reason: str
    instructions: str


class AtomicJanitorAgentInputOutput(BaseModel):
    status: Literal["OK", "ERROR"] = Field(default="ERROR")

    fixed_relationships: Optional[List[_ArchitectAgentRelationship]] = None
    wrong_relationships: Optional[List[AtomicJanitorAgentWrongRelationship]] = None


class TokenInputDetail(BaseModel):
    total: int
    uncached: int
    cached: int
    cache_percentage: float


class TokenOutputDetail(BaseModel):
    total: int
    regular: int
    reasoning: int
    reasoning_percentage: float


class TokenDetail(BaseModel):
    input: TokenInputDetail
    output: TokenOutputDetail
    grand_total: int
    effective_total: int
