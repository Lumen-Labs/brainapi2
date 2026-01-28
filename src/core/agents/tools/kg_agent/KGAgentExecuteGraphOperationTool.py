"""
File: /KGAgentExecuteGraphOperationTool.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 9:26:22 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter


class KGAgentExecuteGraphOperationTool(BaseTool):
    """
    Tool for executing a graph operation.
    """

    name: str = "kg_agent_execute_graph_operation"

    kg_agent: object
    kg: GraphAdapter
    brain_id: str = "default"

    def __init__(
        self, kg_agent, kg: GraphAdapter, database_desc: str, brain_id: str = "default"
    ):
        """
        Initialize the KGAgentExecuteGraphOperationTool with the specified components and configuration.
        
        Parameters:
            kg_agent: The knowledge graph agent to execute operations.
            kg (GraphAdapter): The graph adapter managing the graph database interface.
            database_desc (str): Description of the underlying graph database to include in the tool's description.
            brain_id (str): Identifier for the brain context to use, defaults to "default".
        """
        description: str = (
            "Use this tool to execute any type of graph operation search/edit/query/delete/etc. "
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
        super().__init__(
            kg_agent=kg_agent,
            kg=kg,
            description=formatted_description,
            brain_id=brain_id,
        )

    def _run(self, *args, **kwargs) -> str:
        """
        Execute a graph operation using a query extracted from the provided arguments and return the KG adapter's response.
        
        The function accepts several argument shapes to locate the query:
        - positional: first positional argument is used as the query string.
        - kwargs 'args' as dict: use the value of the 'query' key.
        - kwargs 'args' as list: use the first element if it's a string, or the first element's 'query' value if it's a dict.
        
        Returns:
            The result returned by the graph adapter's execute_operation for the resolved query, or the string "No query provided in the arguments or kwargs" if no query could be found.
        """
        _query = ""
        if len(args) > 0:
            _query = args[0]

        if len(kwargs) > 0:
            args_query = kwargs.get("args", {})
            if isinstance(args_query, dict):
                _query = args_query.get("query", "")
            elif isinstance(args_query, list) and len(args_query) > 0:
                first_arg = args_query[0]
                if isinstance(first_arg, str):
                    _query = first_arg
                elif isinstance(first_arg, dict):
                    _query = first_arg.get("query", "")

        if len(_query) == 0:
            return "No query provided in the arguments or kwargs"

        print(
            "[DEBUG (kg_agent_execute_graph_operation)]: Executing query: ",
            _query,
        )

        response = self.kg.execute_operation(_query, brain_id=self.brain_id)

        print(
            "[DEBUG (kg_agent_execute_graph_operation)]: Response: ",
            response,
        )

        return response