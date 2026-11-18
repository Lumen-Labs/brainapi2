import json

MCP_TOOL_SPECS: list[dict[str, object]] = [
    {
        "name": "get_search_operation_instructions",
        "description": "Returns instructions for searching the knowledge graph (backend type, recommended workflow, parameter reference).",
        "parameters": {"message": "str — optional context for the instructions request"},
    },
    {
        "name": "list_brains",
        "description": "Lists brains/memory stores available to the current token.",
        "parameters": {},
    },
    {
        "name": "search_semantically",
        "description": "Semantic search when node UUIDs or exact names are unknown.",
        "parameters": {
            "query": "str",
            "brain_id": "str — use the active brain_id for this session",
        },
    },
    {
        "name": "traverse_graph",
        "description": "Multi-hop graph traversal from a start node. The result includes hops and invalidated hops for superseded relationships.",
        "parameters": {
            "brain_id": "str",
            "start_uuid": "str (optional if start_name + start_labels provided)",
            "start_name": "str",
            "start_labels": "list[str] (optional)",
            "depth": "int 1-5, default 2",
            "rel_types": "list[str] (optional)",
            "node_labels": "list[str] (optional)",
            "direction": "in | out | both, default both",
            "limit": "int 1-100, default 50",
        },
    },
    {
        "name": "list_changelogs",
        "description": "Lists graph changelog entries, newest first. Filter with query_text or changelog types.",
        "parameters": {
            "brain_id": "str",
            "query_text": "str (optional)",
            "types": "list[str] (optional): relationship_created, relationship_deprecated, node_properties_updated, relationship_properties_updated",
            "limit": "int 1-50, default 10",
            "skip": "int, default 0",
        },
    },
    {
        "name": "get_changelog",
        "description": "Reads one graph changelog entry by id.",
        "parameters": {
            "brain_id": "str",
            "id": "str",
        },
    },
    {
        "name": "list_source_versions",
        "description": "Reads the current copy and previous copies of a source. Pass source_key, or a query when the key is unknown.",
        "parameters": {
            "brain_id": "str",
            "source_key": "str (optional if query is set)",
            "query": "str (optional if source_key is set)",
            "limit": "int 1-20, default 10",
            "skip": "int, default 0",
        },
    },
    {
        "name": "search_memory",
        "description": "Ad-hoc read-only graph query when traverse_graph is not sufficient.",
        "parameters": {
            "db_query": "str — read-only Cypher (Neo4j) or SQL SELECT/WITH (PostgreSQL)",
            "brain_id": "str",
        },
    },
]


HISTORY_NAVIGATION_GUIDE = (
    "traverse_graph returns hops plus invalidated hops whose relationship properties "
    "include invalid_at or deprecated. "
    "list_changelogs and get_changelog read prior graph changes. "
    "list_source_versions reads the current copy and previous copies of a source."
)


def _format_tool_specs() -> str:
    lines: list[str] = []
    for spec in MCP_TOOL_SPECS:
        params = spec.get("parameters") or {}
        if params:
            param_text = json.dumps(params, indent=2)
        else:
            param_text = "{}"
        lines.append(
            f"- {spec['name']}: {spec['description']}\n  Parameters: {param_text}"
        )
    return "\n".join(lines)


def build_mcp_tools_instructions(brain_id: str) -> str:
    tools_block = _format_tool_specs()
    return f"""You can query the BrainAPI knowledge graph using MCP tools.

Active brain_id for this session: {brain_id}
Always pass this brain_id to tools that require it unless the user explicitly names another brain.

To call a tool, respond with ONLY a JSON object (no markdown fences, no extra text):
{{"tool_name": "<tool_name>", "tool_input": {{<parameters>}}}}

Available tools:
{tools_block}

Recommended workflow:
1. Call get_search_operation_instructions when you are unsure how to search.
2. search_semantically — when you do not know node UUIDs or exact names.
3. traverse_graph — multi-hop exploration (preferred over hand-written queries). {HISTORY_NAVIGATION_GUIDE}
4. search_memory — ad-hoc read queries not covered by traverse_graph.

After each tool call you will receive a tool result message. You may call another tool or reply to the user.
When you have enough context to answer, respond in natural language without a tool call JSON object.
"""
