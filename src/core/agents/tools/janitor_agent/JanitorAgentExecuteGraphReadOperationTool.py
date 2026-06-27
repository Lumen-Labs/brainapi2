"""
File: /JanitorAgentExecuteGraphReadOperationTool.py
Created Date: Wednesday December 24th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 24th 2025 9:26:30 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
import re

from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter

_UNION_SPLIT_PATTERN = re.compile(r"\bUNION\s+ALL\b|\bUNION\b", re.IGNORECASE)


def _split_union_query(query: str) -> list[str]:
    parts = _UNION_SPLIT_PATTERN.split(query)
    return [part.strip() for part in parts if part.strip()]


def _merge_graph_read_results(results: list[str]) -> str:
    merged_records = []
    keys = None
    errors = []

    for result in results:
        if result.startswith("Error executing graph operation"):
            errors.append(result)
            continue
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            merged_records.append({"raw": result})
            continue
        if isinstance(payload, dict) and "records" in payload:
            merged_records.extend(payload.get("records", []))
            if keys is None and payload.get("keys"):
                keys = payload.get("keys")
        else:
            merged_records.append(payload)

    if not merged_records and errors:
        return errors[0]

    payload = {
        "records": merged_records[:20],
        "truncated": len(merged_records) > 20,
    }
    if keys is not None:
        payload["keys"] = keys
    if errors:
        payload["partial_errors"] = errors
    return json.dumps(payload, default=str)


class JanitorAgentExecuteGraphReadOperationTool(BaseTool):
    """
    Tool for executing a graph read only operation.
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
            "Prefer `search_entities` for entity lookup. "
            "Use this tool for ad-hoc graph reads when search_entities is not enough. "
            "The query should be a valid graph read only operation depending on the graph database type. "
            "{database_desc}. "
            "For UNION queries, every branch must RETURN the same column aliases "
            "(e.g. RETURN n.uuid AS uuid, n.name AS name). "
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
            arg_val = args[0]
            if isinstance(arg_val, dict):
                _query = arg_val.get("query", "")
            else:
                _query = arg_val
        else:
            _query = kwargs.get("query", "")

        if len(_query) == 0:
            return "No query provided in the arguments or kwargs"

        if _UNION_SPLIT_PATTERN.search(_query):
            subqueries = _split_union_query(_query)
            if len(subqueries) > 1:
                results = [
                    self.kg.execute_operation(subquery, brain_id=self.brain_id)
                    for subquery in subqueries
                ]
                return _merge_graph_read_results(results)

        return self.kg.execute_operation(_query, brain_id=self.brain_id)
