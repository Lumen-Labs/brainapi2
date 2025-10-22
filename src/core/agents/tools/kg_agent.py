"""
File: /kg_agent.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 3:36:26 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
import uuid
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Triple
from src.services.api.constants.tool_schemas import NODE_SCHEMA, TRIPLE_SCHEMA


class KGAgentExecuteGraphOperationTool(BaseTool):
    """
    Tool for executing a graph operation.
    """

    name: str = "kg_agent_execute_graph_operation"

    kg_agent: object
    kg: GraphAdapter

    def __init__(self, kg_agent, kg: GraphAdapter, database_desc: str):
        description: str = (
            "Tool for executing search operations on the knowledge graph. "
            "Use this tool to search for information and existing nodes in the knowledge graph. "
            "The query should be a valid graph operation depending on the graph database type."
            "The query should be a valid JSON object with a 'query' key. "
            "{database_desc}."
            "If you get an error, try again after fixing your query and don't give up."
        )
        # description: str = (
        #     "Tool for executing any type of graph operation search/edit/query/delete/etc. "
        #     "The query should be a valid graph operation depending on the graph database type."
        #     "The query should be a valid JSON object with a 'query' key. "
        #     "{database_desc}."
        #     "If you get an error, try again after fixing your query and don't give up."
        # )
        formatted_description = description.format(database_desc=database_desc)
        super().__init__(kg_agent=kg_agent, kg=kg, description=formatted_description)

    def _run(self, *args, **kwargs) -> str:
        _query = ""
        if len(args) > 0:
            _query = args[0]

        if len(kwargs) > 0:
            args_query = kwargs.get("args", {})
            if isinstance(args_query, dict):
                _query = args_query.get("query", "")
            elif isinstance(args_query, list):
                _query = args_query[0].get("query", "")

        if len(_query) == 0:
            return "No query provided in the arguments or kwargs"

        response = self.kg.execute_operation(_query)

        return response


class KGAgentAddNodesTool(BaseTool):
    """
    Tool for adding nodes to the knowledge graph.
    """

    name: str = "kg_agent_add_nodes"

    kg_agent: object
    kg: GraphAdapter
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None

    args_schema: dict = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": NODE_SCHEMA,
            },
        },
        "required": ["nodes"],
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        description: str = (
            "Tool specifically for adding nodes to the knowledge graph. "
            "This tool will return the nodes that were added to the knowledge graph."
            "Input should be a valid JSON object with a 'nodes' key. "
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=description,
            identification_params=identification_params or {},
            metadata=metadata or {},
        )

    def _run(self, *args, **kwargs) -> str:
        nodes = [
            Node(**node, uuid=str(uuid.uuid4())) for node in kwargs.get("nodes", [])
        ]
        self.kg.add_nodes(nodes, self.identification_params, self.metadata)
        return "Nodes added successfully"


class KGAgentAddTripletsTool(BaseTool):
    """
    Tool for adding triplets to the knowledge graph.
    """

    name: str = "kg_agent_add_triplets"
    kg_agent: object
    kg: GraphAdapter
    identification_params: Optional[dict] = None
    metadata: Optional[dict] = None

    args_schema: dict = {
        "type": "object",
        "properties": {
            "triplets": {
                "type": "array",
                "description": ("The triplets to add to the knowledge graph."),
                "items": TRIPLE_SCHEMA,
            },
        },
    }

    def __init__(
        self,
        kg_agent,
        kg: GraphAdapter,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        description: str = (
            "This tool is used to register triplets into the knowledge graph. "
            "Returns the triplets that were added to the knowledge graph. "
        )
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=description,
            identification_params=identification_params or {},
            metadata=metadata or {},
        )

    def _run(self, *args, **kwargs) -> str:
        triplets = []
        for triplet_data in kwargs.get("triplets", []):
            subject = Node(**triplet_data["subject"])
            object_node = Node(**triplet_data["object"])

            triplet = Triple(
                subject=subject, predicate=triplet_data["predicate"], object=object_node
            )
            triplets.append(triplet)

        for triplet in triplets:
            print(
                f"{triplet.subject.name} - {triplet.predicate} -> {triplet.object.name}"
            )
            self.kg.add_nodes(
                [triplet.subject, triplet.object],
                self.identification_params,
                self.metadata,
            )
        return f"Triplets added successfully: {triplets}"


class KGAgentAddRelationshipsTool(BaseTool):
    pass


class KGAgentUpdateNodesTool(BaseTool):
    pass


class KGAgentDeleteNodesTool(BaseTool):
    pass


class KGAgentDeleteRelationshipsTool(BaseTool):
    pass
