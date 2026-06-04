import asyncio
import json
from typing import Any

from src.services.mcp.prompt import MCP_TOOL_SPECS

_MCP_TOOL_NAMES = {spec["name"] for spec in MCP_TOOL_SPECS}


def _normalize_tool_input(tool_input: Any) -> dict[str, Any]:
    if tool_input is None:
        return {}
    if isinstance(tool_input, dict):
        return tool_input
    if isinstance(tool_input, str):
        raw = tool_input.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"message": raw}
    return {"input": tool_input}


async def execute_mcp_tool(
    tool_name: str,
    tool_input: Any,
    *,
    brain_pat: str | None,
) -> str:
    if tool_name not in _MCP_TOOL_NAMES:
        return f"Error: unknown tool '{tool_name}'"

    from src.services.mcp.main import (
        auth_token_var,
        get_search_operation_instructions,
        list_brains,
        search_memory,
        search_semantically,
        traverse_graph,
    )

    params = _normalize_tool_input(tool_input)
    token = auth_token_var.set(brain_pat)
    try:
        if tool_name == "get_search_operation_instructions":
            message = str(params.get("message") or "")
            result = get_search_operation_instructions(message)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "list_brains":
            result = await list_brains()
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "search_semantically":
            result = await search_semantically(
                query=str(params.get("query") or ""),
                brain_id=str(params.get("brain_id") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "traverse_graph":
            result = await traverse_graph(
                brain_id=str(params.get("brain_id") or ""),
                start_uuid=str(params.get("start_uuid") or ""),
                start_name=str(params.get("start_name") or ""),
                start_labels=params.get("start_labels"),
                depth=int(params.get("depth") or 2),
                rel_types=params.get("rel_types"),
                node_labels=params.get("node_labels"),
                direction=str(params.get("direction") or "both"),
                limit=int(params.get("limit") or 50),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "search_memory":
            result = await search_memory(
                db_query=str(params.get("db_query") or ""),
                brain_id=str(params.get("brain_id") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        return f"Error: tool '{tool_name}' is not implemented"
    except Exception as exc:
        return f"Error executing tool '{tool_name}': {exc}"
    finally:
        auth_token_var.reset(token)
