"""
File: /KGAgentRemoveRelationshipTool.py
Project: kg_agent
Created Date: Friday January 30th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday January 30th 2026 9:52:56 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.embeddings import VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.constants.kg import NodeDict, PredicateDict


class KGAgentRemoveRelationshipTool(BaseTool):
    """
    Tool for removing a relationship from the knowledge graph.
    """

    name: str = "kg_agent_remove_relationship"

    kg_agent: object
    kg: GraphAdapter
    vector_store: VectorStoreAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "relationship": {
                "type": "string",
                "description": "The UUID of the relationship to remove.",
            },
            "tail": {
                "type": "object",
                "properties": {
                    "uuid": {
                        "type": "string",
                        "description": "The UUID of the tail node.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the tail node.",
                    },
                    "labels": {
                        "type": "array",
                        "description": "The labels of the tail node.",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "description": "The tail node of the relationship to remove. Can contain the UUID of the tail node, OR the name AND labels of the tail node.",
            },
            "head": {
                "type": "object",
                "properties": {
                    "uuid": {
                        "type": "string",
                        "description": "The UUID of the head node.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the head node.",
                    },
                    "labels": {
                        "type": "array",
                        "description": "The labels of the head node.",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "description": "The head node of the relationship to remove. Can contain the UUID of the head node, OR the name AND labels of the head node.",
            },
            "relationship_name": {
                "type": "string",
                "description": "The name of the relationship to remove to be provided along with the tail and head nodes.",
            },
        },
        "description": "The relationship to remove. Can contain the UUID of the relationship, OR the tail and head nodes and the relationship name.",
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        brain_id: str = "default",
    ):
        """
        Initialize the KGAgentRemoveRelationshipTool with the specified components and configuration.

        Parameters:
            kg_agent: The knowledge graph agent to execute operations.
            kg (GraphAdapter): The graph adapter managing the graph database interface.
            vector_store (VectorStoreAdapter): The vector store adapter managing the vector store interface.
            brain_id (str): Identifier for the brain context to use, defaults to "default".
        """
        description: str = (
            "Use this tool to remove a relationship from the knowledge graph. "
            "The relationship should be a valid relationship in the knowledge graph. "
            "The argument must be a valid JSON object that can contain a 'uuid' key OR a 'tail' and 'head' keys and the 'relationship_name' key. "
            "The 'tail' and 'tip' parameters can contain the UUID of the tail and tip nodes, or the name and labels of the tail and tip nodes. "
        )

        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            vector_store=vector_store,
            description=description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Remove a relationship from the knowledge graph.
        """
        _rel_uuid: str = None
        _tail: NodeDict = None
        _head: NodeDict = None
        _relationship_name: str = None
        print(f"[DEBUG (kg_agent_remove_relationship)]: kwargs: {kwargs} args: {args}")
        if len(kwargs) > 0:
            args_uuid = kwargs.get("args", {})
            if isinstance(args_uuid, dict):
                if "uuid" in args_uuid:
                    _rel_uuid = args_uuid.get("uuid", "")
                if "tail" in args_uuid and "head" in args_uuid:
                    _tail = args_uuid.get("tail", {})
                    _head = args_uuid.get("head", {})
                if "relationship_name" in args_uuid:
                    _relationship_name = args_uuid.get("relationship_name", "")
            elif isinstance(args_uuid, list) and len(args_uuid) > 0:
                first_arg = args_uuid[0]
                if isinstance(first_arg, dict):
                    if "uuid" in first_arg:
                        _rel_uuid = first_arg.get("uuid", "")
                    if "tail" in first_arg and "head" in first_arg:
                        _tail = first_arg.get("tail", {})
                        _head = first_arg.get("head", {})
                    if "relationship_name" in first_arg:
                        _relationship_name = first_arg.get("relationship_name", "")

        print(
            f"[DEBUG (kg_agent_remove_relationship)]: Removing relationship: {_rel_uuid} - {_tail} - {_head} - {_relationship_name}"
        )
        try:
            removed_relationships = self.kg.remove_relationships(
                relationships=[
                    (
                        _tail,
                        PredicateDict(
                            {
                                **({"uuid": _rel_uuid} if _rel_uuid else {}),
                                **(
                                    {"name": _relationship_name}
                                    if _relationship_name
                                    else {}
                                ),
                            }
                        ),
                        _head,
                    )
                ],
                brain_id=self.brain_id,
            )
        except Exception as e:
            print(
                f"[DEBUG (kg_agent_remove_relationship)]: Error removing relationship: {e}"
            )
            return f"Error removing relationship: {e}"

        if len(removed_relationships) > 0:
            v_ids = [r.properties.get("v_id") for _, r, _ in removed_relationships if r.properties.get("v_id")]
            if len(v_ids) > 0:
                self.vector_store.remove_vectors(
                    v_ids, store="relationships", brain_id=self.brain_id
                )

        # Returning success in any case to prevent the kg_agent to iterate across ghost relationships
        return "Operation completed successfully"
