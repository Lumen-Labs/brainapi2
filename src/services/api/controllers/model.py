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


async def add_nodes(
    nodes: list[dict],
    brain_id: str,
    identification_params: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> list[Node] | str:
    """
    Create and add multiple Node objects to the graph.

    Each input dict is converted into a Node (a new UUID is generated for each) and then added to the graph via the graph adapter.

    Parameters:
        nodes (list[dict]): List of node data dicts. Expected keys include `name`, optional `labels` (list), optional `description`, and optional `properties` (dict).
        brain_id (str): Identifier for the target brain/graph.
        identification_params (dict, optional): Parameters used to identify existing nodes for deduplication or matching; passed through to the graph adapter.
        metadata (dict, optional): Additional metadata forwarded to the graph adapter.

    Returns:
        list[Node] | str: The list of added Node objects, or a string result as returned by the graph adapter.
    """
    from src.constants.kg import Node
    import uuid as uuid_lib

    node_objects = []
    for node_data in nodes:
        name = node_data.get("name")
        if not name:
            raise ValueError("Node name is required and cannot be empty")

        properties = node_data.get("properties")
        if properties is None:
            properties = {}

        node_objects.append(
            Node(
                uuid=str(uuid_lib.uuid4()),
                name=name,
                labels=node_data.get("labels", []),
                description=node_data.get("description"),
                properties=properties,
            )
        )

    result = await asyncio.to_thread(
        graph_adapter.add_nodes,
        nodes=node_objects,
        brain_id=brain_id,
        identification_params=identification_params,
        metadata=metadata,
    )

    return result


async def update_node(
    uuid: str,
    brain_id: str,
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
    new_labels: Optional[list[str]] = None,
    new_properties: Optional[dict] = None,
    properties_to_remove: Optional[list[str]] = None,
) -> Node | None:
    """
    Update fields of a node in the graph.

    Only the provided parameters will be applied to the node; omitted optional parameters are left unchanged.

    Parameters:
        uuid (str): UUID of the node to update.
        brain_id (str): Identifier of the brain/graph where the node resides.
        new_name (Optional[str]): New name for the node.
        new_description (Optional[str]): New description for the node.
        new_labels (Optional[list[str]]): Labels to assign to the node.
        new_properties (Optional[dict]): Properties to set or update on the node.
        properties_to_remove (Optional[list[str]]): List of property keys to remove from the node.

    Returns:
        Node | None: The updated Node if it exists, `None` if no node with the given UUID was found.
    """
    result = await asyncio.to_thread(
        graph_adapter.update_node,
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
    Create a predicate relationship between two nodes identified by UUIDs within the specified brain.

    Parameters:
        subject_uuid (str): UUID of the subject node.
        predicate_name (str): Name of the predicate (relationship).
        predicate_description (str): Description for the predicate.
        object_uuid (str): UUID of the object node.
        brain_id (str): Identifier of the brain/graph where the relationship will be created.

    Returns:
        str: Adapter result for the created relationship (typically the relationship identifier).
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
    Update properties of an existing relationship identified by its UUID.

    Parameters:
        uuid (str): UUID of the relationship to update.
        brain_id (str): Identifier of the brain/graph scope where the relationship exists.
        new_properties (dict): Properties to add or update on the relationship.
        properties_to_remove (list[str]): List of property names to remove from the relationship.

    Returns:
        Predicate | None: The updated Predicate object if the relationship was found and updated, `None` otherwise.
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
