"""
File: /model.py
Created Date: Saturday December 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from typing import Optional
from src.services.kg_agent.main import graph_adapter
from src.constants.kg import Node, Predicate

async def add_entity(
    name: str,
    brain_id: str,
    labels: list[str],
    description: Optional[str] = None,
    properties: Optional[dict] = None,
    identification_params: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Node | None:
    """
    Add a single entity (node) to the graph.
    """
    result = await asyncio.to_thread(
        graph_adapter.add_entity,
        name=name,
        brain_id=brain_id,
        labels=labels,
        description=description,
        properties=properties,
        identification_params=identification_params,
        metadata=metadata,
    )
    return result


async def update_entity(
    uuid: str,
    brain_id: str,
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
    new_labels: Optional[list[str]] = None,
    new_properties: Optional[dict] = None,
    properties_to_remove: Optional[list[str]] = None,
) -> Node | None:
    """
    Update an entity (node) in the graph.
    """
    result = await asyncio.to_thread(
        graph_adapter.update_entity,
        uuid=uuid,
        brain_id=brain_id,
        new_name=new_name,
        new_description=new_description,
        new_labels=new_labels,
        new_properties=new_properties,
        properties_to_remove=properties_to_remove,
    )
    return result


async def add_relationship(
    subject_uuid: str,
    predicate_name: str,
    predicate_description: str,
    object_uuid: str,
    brain_id: str,
) -> str:
    """
    Add a relationship between two nodes in the graph.
    """
    subject_node = await asyncio.to_thread(
        graph_adapter.get_by_uuid,
        uuid=subject_uuid,
        brain_id=brain_id,
    )
    
    object_node = await asyncio.to_thread(
        graph_adapter.get_by_uuid,
        uuid=object_uuid,
        brain_id=brain_id,
    )
    
    if not subject_node:
        raise ValueError(f"Subject node with UUID {subject_uuid} not found")
    
    if not object_node:
        raise ValueError(f"Object node with UUID {object_uuid} not found")
    
    import uuid as uuid_lib
    predicate = Predicate(
        uuid=str(uuid_lib.uuid4()),
        name=predicate_name,
        description=predicate_description,
    )
    
    result = await asyncio.to_thread(
        graph_adapter.add_relationship,
        subject=subject_node,
        predicate=predicate,
        to_object=object_node,
        brain_id=brain_id,
    )
    
    return result


async def update_relationship(
    uuid: str,
    brain_id: str,
    new_properties: dict,
    properties_to_remove: list[str],
) -> Predicate | None:
    """
    Update a relationship's properties in the graph.
    """
    result = await asyncio.to_thread(
        graph_adapter.update_properties,
        uuid=uuid,
        updating="relationship",
        brain_id=brain_id,
        new_properties=new_properties,
        properties_to_remove=properties_to_remove,
    )
    return result