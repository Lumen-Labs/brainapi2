"""
File: /KGAgentRemoveNodeTool.py
Project: kg_agent
Created Date: Friday January 30th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday January 30th 2026 9:53:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.embeddings import VectorStoreAdapter
from src.adapters.graph import GraphAdapter


class KGAgentRemoveNodeTool(BaseTool):
    """
    Tool for removing a node from the knowledge graph.
    """

    name: str = "kg_agent_remove_node"

    kg_agent: object
    kg: GraphAdapter
    vector_store: VectorStoreAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "The UUID of the node to remove.",
            },
        },
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        brain_id: str = "default",
    ):
        """
        Initialize the KGAgentExecuteGraphOperationTool with the specified components and configuration.

        Parameters:
            kg_agent: The knowledge graph agent to execute operations.
            kg (GraphAdapter): The graph adapter managing the graph database interface.
            vector_store (VectorStoreAdapter): The vector store adapter managing the vector store interface.
            brain_id (str): Identifier for the brain context to use, defaults to "default".
        """
        description: str = (
            "Use this tool to remove a node from the knowledge graph. "
            "The node should be a valid node in the knowledge graph. "
            "The argument should be a valid JSON object with a 'uuid' key. "
            "The 'uuid' key should be the UUID of the node to remove. "
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
        Remove a node from the knowledge graph.

        The function accepts several argument shapes to locate the node:
        - positional: first positional argument is used as the UUID of the node to remove.
        - kwargs 'args' as dict: use the value of the 'uuid' key.
        - kwargs 'args' as list: use the first element if it's a string, or the first element's 'uuid' value if it's a dict.

        Returns:
            The result returned by the graph adapter's execute_operation for the resolved query, or the string "No query provided in the arguments or kwargs" if no query could be found.
        """
        _query = ""
        if len(args) > 0:
            _query = args[0]

        if len(kwargs) > 0:
            args_uuid = kwargs.get("args", {})
            if isinstance(args_uuid, dict):
                _uuid = args_uuid.get("uuid", "")
            elif isinstance(args_uuid, list) and len(args_uuid) > 0:
                first_arg = args_uuid[0]
                if isinstance(first_arg, str):
                    _uuid = first_arg
                elif isinstance(first_arg, dict):
                    _uuid = first_arg.get("uuid", "")

        print(f"[DEBUG (kg_agent_remove_node)]: kwargs: {kwargs} args: {args}")

        if len(_query) == 0:
            return "No UUID provided in the arguments or kwargs"

        print(f"[DEBUG (kg_agent_remove_node)]: Removing node: {_uuid}")
        removed_nodes = self.kg.remove_nodes(uuids=[_uuid], brain_id=self.brain_id)

        if len(removed_nodes) > 0:
            v_ids = [n for n in removed_nodes if n.properties.get("v_id")]
            if len(v_ids) > 0:
                self.vector_store.remove_vectors(
                    v_ids, store="nodes", brain_id=self.brain_id
                )

        # Returning success in any case to prevent the kg_agent to iterate across ghost nodes
        return "Operation completed successfully"
