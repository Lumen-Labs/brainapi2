import asyncio
import json
from typing import Any

from src.services.mcp.prompt import MCP_TOOL_SPECS

_MCP_TOOL_NAMES = {spec["name"] for spec in MCP_TOOL_SPECS}


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, default=str)
    if isinstance(result, dict):
        return json.dumps(result, default=str)
    return json.dumps(result, default=str)


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

    from src.services.mcp.intent_tools import (
        brainapi_check,
        brainapi_correct,
        brainapi_recall,
        brainapi_remember,
    )
    from src.services.mcp.main import (
        auth_token_var,
        conflict_report,
        dismiss_conflict,
        get_changelog,
        get_conflict,
        get_search_operation_instructions,
        list_brains,
        list_changelogs,
        list_conflicts,
        list_source_versions,
        resolve_conflict,
        search_memory,
        search_semantically,
        traverse_graph,
    )

    params = _normalize_tool_input(tool_input)
    token = auth_token_var.set(brain_pat)
    try:
        if tool_name == "brainapi_recall":
            result = await brainapi_recall(
                question=str(params.get("question") or ""),
                as_of=str(params.get("as_of") or ""),
                response_format=str(params.get("response_format") or "concise"),
                include_rejected=bool(params.get("include_rejected") or False),
                limit=int(params.get("limit") or 8),
                skip=int(params.get("skip") or 0),
                brain_id=str(params.get("brain_id") or ""),
            )
            return _stringify_tool_result(result)
        if tool_name == "brainapi_check":
            result = await brainapi_check(
                claim=str(params.get("claim") or ""),
                as_of=str(params.get("as_of") or ""),
                brain_id=str(params.get("brain_id") or ""),
            )
            return _stringify_tool_result(result)
        if tool_name == "brainapi_remember":
            result = await brainapi_remember(
                fact=str(params.get("fact") or ""),
                source=str(params.get("source") or ""),
                valid_from=str(params.get("valid_from") or ""),
                source_key=str(params.get("source_key") or ""),
                brain_id=str(params.get("brain_id") or ""),
            )
            return _stringify_tool_result(result)
        if tool_name == "brainapi_correct":
            result = await brainapi_correct(
                reason=str(params.get("reason") or ""),
                fact_id=str(params.get("fact_id") or ""),
                text=str(params.get("text") or params.get("fact") or ""),
                correction=str(params.get("correction") or ""),
                brain_id=str(params.get("brain_id") or ""),
            )
            return _stringify_tool_result(result)
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
                include_rejected=bool(params.get("include_rejected") or False),
                as_of=str(params.get("as_of") or "") or None,
                include_validity_unknown=bool(
                    params.get("include_validity_unknown") or False
                ),
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
        if tool_name == "list_changelogs":
            result = await list_changelogs(
                brain_id=str(params.get("brain_id") or ""),
                query_text=str(params.get("query_text") or ""),
                types=params.get("types"),
                limit=int(params.get("limit") or 10),
                skip=int(params.get("skip") or 0),
                since=str(params.get("since") or ""),
                until=str(params.get("until") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "get_changelog":
            result = await get_changelog(
                brain_id=str(params.get("brain_id") or ""),
                id=str(params.get("id") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "list_source_versions":
            result = await list_source_versions(
                brain_id=str(params.get("brain_id") or ""),
                source_key=str(params.get("source_key") or ""),
                query=str(params.get("query") or ""),
                limit=int(params.get("limit") or 10),
                skip=int(params.get("skip") or 0),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "list_conflicts":
            result = await list_conflicts(
                brain_id=str(params.get("brain_id") or ""),
                status=str(params.get("status") or ""),
                source=str(params.get("source") or ""),
                subject=str(params.get("subject") or ""),
                since=str(params.get("since") or ""),
                until=str(params.get("until") or ""),
                limit=int(params.get("limit") or 10),
                skip=int(params.get("skip") or 0),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "get_conflict":
            result = await get_conflict(
                brain_id=str(params.get("brain_id") or ""),
                id=str(params.get("id") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "resolve_conflict":
            result = await resolve_conflict(
                brain_id=str(params.get("brain_id") or ""),
                id=str(params.get("id") or ""),
                winning_fact_id=str(params.get("winning_fact_id") or ""),
                notes=str(params.get("notes") or ""),
                method=str(params.get("method") or "manual"),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "dismiss_conflict":
            result = await dismiss_conflict(
                brain_id=str(params.get("brain_id") or ""),
                id=str(params.get("id") or ""),
                notes=str(params.get("notes") or ""),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        if tool_name == "conflict_report":
            result = await conflict_report(
                brain_id=str(params.get("brain_id") or ""),
                status=str(params.get("status") or ""),
                source=str(params.get("source") or ""),
                subject=str(params.get("subject") or ""),
                since=str(params.get("since") or ""),
                until=str(params.get("until") or ""),
                format=str(params.get("format") or "json"),
                limit=int(params.get("limit") or 200),
            )
            return result if isinstance(result, str) else json.dumps(result, default=str)
        return f"Error: tool '{tool_name}' is not implemented"
    except Exception as exc:
        return f"Error executing tool '{tool_name}': {exc}"
    finally:
        auth_token_var.reset(token)
