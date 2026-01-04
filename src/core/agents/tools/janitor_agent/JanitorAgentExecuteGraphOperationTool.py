"""
File: /JanitorAgentExecuteGraphOperationTool.py
Created Date: Friday January 2nd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday January 2nd 2026 11:42:24 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter


class JanitorAgentExecuteGraphOperationTool(BaseTool):
    """
    Tool for executing a graph operation.
    """

    name: str = "execute_graph_read_operation"

    janitor_agent: object
    kg: GraphAdapter
    brain_id: str = "default"

    args_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to execute on the knowledge graph. Should be a valid graph read only operation depending on the graph database type.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        janitor_agent,
        kg: GraphAdapter,
        database_desc: str,
        brain_id: str = "default",
    ):
        description: str = (
            "Tool for executing read only operations on the knowledge graph. "
            "Use this tool to read information from the knowledge graph. "
            "The query should be a valid graph read only operation depending on the graph database type."
            "The query should be a valid JSON object with a 'query' key. "
            "{database_desc}."
            "If you get an error, try again after fixing your query and don't give up. "
            "If you get a result, return the result as a JSON object."
        )
        formatted_description = description.format(database_desc=database_desc)
        super().__init__(
            janitor_agent=janitor_agent,
            kg=kg,
            description=formatted_description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        _query = ""
        if len(args) > 0:
            _query = args[0]
        else:
            _query = kwargs.get("query", "")

        if len(_query) == 0:
            return "No query provided in the arguments or kwargs"

        response = self.kg.execute_operation(_query, brain_id=self.brain_id)

        return response
