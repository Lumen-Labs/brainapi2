import json

MCP_TOOL_SPECS: list[dict[str, object]] = [
    {
        "name": "brainapi_recall",
        "description": "Recall what is true now or as of a date. Use for what's true, as of, what changed, and who said.",
        "parameters": {
            "question": "str",
            "as_of": "str ISO-8601 (optional)",
            "response_format": "concise | detailed, default concise",
            "include_rejected": "bool, default false",
            "limit": "int, default 8",
            "skip": "int, default 0",
            "brain_id": "str (optional; default from OAuth token)",
        },
    },
    {
        "name": "brainapi_check",
        "description": "Check whether a claim is still true. Returns supported, contradicted, superseded, or unknown.",
        "parameters": {
            "claim": "str",
            "as_of": "str ISO-8601 (optional)",
            "brain_id": "str (optional; default from OAuth token)",
        },
    },
    {
        "name": "brainapi_remember",
        "description": "Remember a fact. Writes it unverified. source_key rewrites supersede the previous version.",
        "parameters": {
            "fact": "str",
            "source": "str — document name or URL",
            "valid_from": "str ISO-8601 (optional)",
            "source_key": "str (optional)",
            "brain_id": "str (optional; default from OAuth token)",
        },
    },
    {
        "name": "brainapi_correct",
        "description": "Correct a stored fact without deleting history. Pass fact_id or text, a reason, and optional correction.",
        "parameters": {
            "reason": "str",
            "fact_id": "str (optional if text is set)",
            "text": "str (optional if fact_id is set)",
            "correction": "str (optional; omit to reject)",
            "brain_id": "str (optional; default from OAuth token)",
        },
    },
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
            "include_rejected": "bool (optional, default false)",
            "as_of": "str ISO-8601 (optional): knowledge state valid at that instant",
            "include_validity_unknown": "bool (optional, default false): include undated facts in validity_unknown",
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
        "description": "Lists graph changelog entries, newest first. Filter with query_text, changelog types, or a since/until window of added, superseded, and invalidated facts.",
        "parameters": {
            "brain_id": "str",
            "query_text": "str (optional)",
            "types": "list[str] (optional): relationship_created, relationship_deprecated, node_properties_updated, relationship_properties_updated",
            "limit": "int 1-50, default 10",
            "skip": "int, default 0",
            "since": "str ISO-8601 (optional, inclusive)",
            "until": "str ISO-8601 (optional, exclusive)",
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
        "name": "list_conflicts",
        "description": "Lists cross-source fact conflicts. Filter by status, source_key, subject, or detected_at window.",
        "parameters": {
            "brain_id": "str",
            "status": "open | resolved | dismissed (optional)",
            "source": "str exact source_key (text:{filename}, text:paste, or caller source_id) or a substring of key/document/quote (optional)",
            "subject": "str subject name (optional)",
            "since": "str ISO-8601 (optional)",
            "until": "str ISO-8601 (optional)",
            "limit": "int 1-50, default 10",
            "skip": "int, default 0",
        },
    },
    {
        "name": "get_conflict",
        "description": "Reads one conflict with full provenance for both sides.",
        "parameters": {
            "brain_id": "str",
            "id": "str",
        },
    },
    {
        "name": "resolve_conflict",
        "description": "Resolves a conflict. Pass winning_fact_id when one side is chosen; method=verified_side verifies the winner and rejects the other side via the fact verification API.",
        "parameters": {
            "brain_id": "str",
            "id": "str",
            "winning_fact_id": "str (optional)",
            "notes": "str (optional)",
            "method": "manual | verified_side, default manual",
        },
    },
    {
        "name": "dismiss_conflict",
        "description": "Dismisses a conflict without choosing a winning fact.",
        "parameters": {
            "brain_id": "str",
            "id": "str",
            "notes": "str (optional)",
        },
    },
    {
        "name": "reopen_conflict",
        "description": "Reopens a dismissed or resolved conflict so it is open again. Does not change fact verification state.",
        "parameters": {
            "brain_id": "str",
            "id": "str",
            "notes": "str (optional)",
        },
    },
    {
        "name": "conflict_report",
        "description": "Conflict report with summary counts and per-conflict details. format=json or markdown.",
        "parameters": {
            "brain_id": "str",
            "status": "open | resolved | dismissed (optional)",
            "source": "str (optional)",
            "subject": "str (optional)",
            "since": "str ISO-8601 (optional)",
            "until": "str ISO-8601 (optional)",
            "format": "json | markdown, default json",
            "limit": "int 1-200, default 200",
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
    "list_changelogs and get_changelog read prior graph changes; pass since and until for the knowledge-state diff. "
    "list_source_versions reads the current copy and previous copies of a source. "
    "list_conflicts, get_conflict, resolve_conflict, dismiss_conflict, reopen_conflict, and conflict_report cover cross-source contradictions."
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
brainapi_recall, brainapi_check, brainapi_remember, and brainapi_correct use the token's default brain; pass brain_id only when the token can access more than one. Graph tools that list brain_id still need it.

To call a tool, respond with ONLY a JSON object (no markdown fences, no extra text):
{{"tool_name": "<tool_name>", "tool_input": {{<parameters>}}}}

Available tools:
{tools_block}

Recommended workflow:
1. brainapi_recall — what's true now or as of a date.
2. brainapi_check — is a claim still true.
3. brainapi_remember / brainapi_correct — write or correct a fact.
4. Call get_search_operation_instructions when you need graph query details.
5. search_semantically / traverse_graph / search_memory — power-user graph reads. {HISTORY_NAVIGATION_GUIDE}

After each tool call you will receive a tool result message. You may call another tool or reply to the user.
When you have enough context to answer, respond in natural language without a tool call JSON object.
"""
