"""
File: /main.py
Project: mcp
Created Date: Tuesday February 10th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 22nd 2026 5:21:48 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
import html
import json
import os
from contextvars import ContextVar
from typing import Any

from mcp.server import FastMCP
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from pydantic import AnyHttpUrl, AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from src.core.instances import (
    data_adapter,
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.core.verification.facts import verification_scope
from src.core.verification.service import (
    FactNotFound,
    apply_verification_action,
    get_fact_record,
    list_fact_records,
    require_state,
)
from src.services.mcp.intent_tools import register_intent_tools
from src.services.mcp.oauth_provider import BrainapiMcpOAuthProvider
from src.services.mcp.prompt import HISTORY_NAVIGATION_GUIDE
from src.services.mcp.utils import guard_brainpat
from src.services.mcp.conflicts import (
    conflict_report_tool,
    dismiss_conflict_tool,
    read_conflict,
    read_conflicts,
    resolve_conflict_tool,
)
from src.services.mcp.versions import (
    invalidated_hops,
    read_changelog,
    read_changelogs,
    read_source_versions,
)
from src.utils.vector_search import VectorSearchFacade

auth_token_var: ContextVar[str | None] = ContextVar("auth_token", default=None)
vector_search = VectorSearchFacade(vector_store_adapter)

_oauth_issuer = os.getenv("MCP_OAUTH_ISSUER_URL", "").strip()
_oauth_resource = os.getenv("MCP_RESOURCE_SERVER_URL", "").strip()
if _oauth_issuer and not _oauth_resource:
    _oauth_resource = _oauth_issuer.rstrip("/") + "/mcp"

_oauth_scopes = [
    s for s in os.getenv("MCP_OAUTH_SCOPES", "brainapi").strip().split() if s
]
if not _oauth_scopes:
    _oauth_scopes = ["brainapi"]

_access_ttl = int(os.getenv("MCP_OAUTH_ACCESS_TOKEN_TTL", "3600"))
_refresh_ttl_raw = os.getenv("MCP_OAUTH_REFRESH_TOKEN_TTL", "").strip()
_refresh_ttl = int(_refresh_ttl_raw) if _refresh_ttl_raw else None
_code_ttl = int(os.getenv("MCP_OAUTH_AUTH_CODE_TTL", "600"))

oauth_provider: BrainapiMcpOAuthProvider | None = None
if _oauth_issuer:
    oauth_provider = BrainapiMcpOAuthProvider(
        issuer_url=_oauth_issuer,
        resource_server_url=_oauth_resource,
        valid_scopes=_oauth_scopes,
        access_token_ttl_seconds=_access_ttl,
        refresh_token_ttl_seconds=_refresh_ttl,
        auth_code_ttl_seconds=_code_ttl,
    )
    _doc_url = os.getenv("MCP_OAUTH_SERVICE_DOCUMENTATION_URL", "").strip()
    mcp = FastMCP(
        "brainapi-mcp",
        stateless_http=True,
        host="0.0.0.0",
        auth_server_provider=oauth_provider,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(_oauth_issuer),
            resource_server_url=AnyHttpUrl(_oauth_resource),
            service_documentation_url=AnyHttpUrl(_doc_url) if _doc_url else None,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=_oauth_scopes,
                default_scopes=_oauth_scopes,
            ),
        ),
    )
else:
    mcp = FastMCP("brainapi-mcp", stateless_http=True, host="0.0.0.0")

register_intent_tools(mcp)


if oauth_provider:

    async def _mcp_oauth_consent(request: Request) -> HTMLResponse | RedirectResponse:
        if request.method == "GET":
            q = request.query_params
            client_id = q.get("client_id") or ""
            redirect_uri = q.get("redirect_uri") or ""
            code_challenge = q.get("code_challenge") or ""
            scope = q.get("scope") or " ".join(_oauth_scopes)
            resource = q.get("resource") or ""
            state = q.get("state") or ""
            redirect_uri_provided = q.get("redirect_uri_provided_explicitly") or "1"
            if not (client_id and redirect_uri and code_challenge):
                return HTMLResponse("Missing OAuth parameters", status_code=400)
            client = await oauth_provider.get_client(client_id)
            if not client:
                return HTMLResponse("Unknown client_id", status_code=400)
            esc = html.escape
            form = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Authorize BrainAPI MCP</title></head>
<body>
<h1>Connect to BrainAPI</h1>
<p>Enter your BrainPAT (personal access token for your brain). This authorizes the MCP client to act with that token.</p>
<form method="post" action="/mcp-oauth/consent">
<input type="hidden" name="client_id" value="{esc(client_id)}">
<input type="hidden" name="redirect_uri" value="{esc(redirect_uri)}">
<input type="hidden" name="code_challenge" value="{esc(code_challenge)}">
<input type="hidden" name="scope" value="{esc(scope)}">
<input type="hidden" name="resource" value="{esc(resource)}">
<input type="hidden" name="state" value="{esc(state)}">
<input type="hidden" name="redirect_uri_provided_explicitly" value="{redirect_uri_provided}">
<label for="brainpat">BrainPAT</label>
<input id="brainpat" name="brainpat" type="password" autocomplete="off" required style="width:100%;max-width:40em">
<button type="submit">Authorize</button>
</form>
</body></html>"""
            return HTMLResponse(form)

        form = await request.form()
        client_id = str(form.get("client_id") or "")
        redirect_uri = str(form.get("redirect_uri") or "")
        code_challenge = str(form.get("code_challenge") or "")
        scope_str = str(form.get("scope") or "")
        resource = str(form.get("resource") or "") or None
        state = str(form.get("state") or "") or None
        redirect_uri_provided_explicitly = str(
            form.get("redirect_uri_provided_explicitly") or "1"
        ) == "1"
        brainpat = str(form.get("brainpat") or "").strip()
        if not (client_id and redirect_uri and code_challenge and brainpat):
            return HTMLResponse("Missing fields", status_code=400)
        client = await oauth_provider.get_client(client_id)
        if not client:
            return HTMLResponse("Unknown client", status_code=400)
        if guard_brainpat(brainpat) is False:
            return HTMLResponse("Invalid BrainPAT", status_code=401)
        scopes = scope_str.split() if scope_str.strip() else list(_oauth_scopes)
        for s in scopes:
            if s not in _oauth_scopes:
                return HTMLResponse("Invalid scope", status_code=400)
        try:
            ru = AnyUrl(redirect_uri)
            ru = client.validate_redirect_uri(ru)
        except Exception:
            return HTMLResponse("Invalid redirect_uri", status_code=400)
        code = oauth_provider.issue_auth_code(
            client_id=client_id,
            redirect_uri=ru,
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            code_challenge=code_challenge,
            scopes=scopes,
            resource=resource,
            state=state,
            brainpat=brainpat,
        )
        url = oauth_provider.redirect_after_consent(
            redirect_uri=redirect_uri, code=code, state=state
        )
        return RedirectResponse(url, status_code=302)

    mcp.custom_route("/mcp-oauth/consent", methods=["GET", "POST"])(_mcp_oauth_consent)


@mcp.tool()
def get_search_operation_instructions(message: str) -> str:
    """
    This tool will provide instructions on how to use the search_memory tool.
    """
    backend_type = graph_adapter.graphdb_type
    backend_desc = graph_adapter.graphdb_description
    return f"""
    Graph backend: {backend_type}
    {backend_desc}

    Recommended workflow:
    1. brainapi_recall — what's true now or as of a date
    2. brainapi_check — is a claim still true
    3. brainapi_remember / brainapi_correct — write or correct a fact
    4. search_semantically — when you do not know node UUIDs or exact names
    5. traverse_graph — multi-hop expansion with filters (preferred over hand-written queries)
    6. search_memory — ad-hoc read queries not covered by traverse_graph

    {HISTORY_NAVIGATION_GUIDE}

    traverse_graph parameters:
    - brain_id (required)
    - start_uuid OR (start_name + start_labels)
    - depth (1-5, default 2)
    - rel_types (optional list of relationship type names)
    - node_labels (optional list of node label filters)
    - direction: in | out | both (default both)
    - limit (1-100, default 50)

    search_memory parameters:
    - db_query: read-only graph query ({("SQL SELECT/WITH" if "postgresql" in backend_type else "Cypher")})
    - brain_id: the brain to search in
    """


def _traverse_graph_sync(
    brain_id: str,
    start_uuid: str,
    start_name: str,
    start_labels: list[str] | None,
    depth: int,
    rel_types: list[str] | None,
    node_labels: list[str] | None,
    direction: str,
    limit: int,
    include_rejected: bool,
) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        if direction not in {"in", "out", "both"}:
            return "Error: direction must be in, out, or both"
        with verification_scope(include_rejected=include_rejected):
            result = graph_adapter.traverse_graph(
                brain_id=brain_id,
                start_uuid=start_uuid or None,
                start_name=start_name or None,
                start_labels=start_labels,
                depth=depth,
                rel_types=rel_types,
                node_labels=node_labels,
                direction=direction,
                limit=limit,
            )
        if isinstance(result, dict) and isinstance(result.get("hops"), list):
            result["invalidated"] = invalidated_hops(result["hops"])
        return json.dumps(result, default=str)
    except Exception as e:
        return f"Error traversing graph: {e}"


@mcp.tool()
async def traverse_graph(
    brain_id: str,
    start_uuid: str = "",
    start_name: str = "",
    start_labels: list[str] | None = None,
    depth: int = 2,
    rel_types: list[str] | None = None,
    node_labels: list[str] | None = None,
    direction: str = "both",
    limit: int = 50,
    include_rejected: bool = False,
) -> Any:
    """
    Traverse the knowledge graph from a starting node up to N hops with optional filters.
    Prefer this over search_memory for multi-hop exploration.

    The result keeps every hop and also lists superseded relationships in invalidated.
    Provide start_uuid, or both start_name and start_labels to identify the start node.
    Rejected facts are excluded unless include_rejected is true.
    """
    return await asyncio.to_thread(
        _traverse_graph_sync,
        brain_id,
        start_uuid,
        start_name,
        start_labels,
        depth,
        rel_types,
        node_labels,
        direction,
        limit,
        include_rejected,
    )


def _search_memory_sync(db_query: str, brain_id: str) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        return graph_adapter.execute_operation(db_query, brain_id)
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
async def search_memory(db_query: str, brain_id: str) -> Any:
    """
    Search the brain for memories and information.
    This tool will search into the knowledge graph using the active graph backend query language.

    Input must be a JSON object with the following fields:
    - db_query: str: read-only graph query (Cypher on Neo4j, SQL on PostgreSQL).
      Neo4j runs this in a READ transaction on the selected brain only, with a
      timeout and result cap. Writes, DELETE, USE, CALL dbms.*, and database
      admin are rejected.
    - brain_id: str: the brain to search in.
    """
    return await asyncio.to_thread(_search_memory_sync, db_query, brain_id)


def _search_semantically_sync(query: str, brain_id: str, include_rejected: bool) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        query_embedding = embeddings_adapter.embed_text(query)
        data_vectors = vector_search.search_nodes(
            query_embedding.embeddings,
            brain_id=brain_id,
            k=5,
        )
        with verification_scope(include_rejected=include_rejected):
            triplets = graph_adapter.get_event_centric_neighbors(
                [v.metadata.get("uuid") for v in data_vectors], brain_id=brain_id
            )
        return triplets
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
async def search_semantically(
    query: str, brain_id: str, include_rejected: bool = False
) -> Any:
    """
    Search information semantically, given a query and a brain to search in,
    this tool will return a list of graph nodes that are semantically related to the query.

    This tool is useful to search into the graph for information without knowing names or labels.

    Input must be a JSON object with the following fields:
    - query: str: the query to search for.
    - brain_id: str: the brain to search in.
    - include_rejected: bool: include expert-rejected facts (default false).
    """
    return await asyncio.to_thread(
        _search_semantically_sync, query, brain_id, include_rejected
    )


def _list_brains_sync() -> list[str] | str:
    brain_key = guard_brainpat(auth_token_var.get())
    if not brain_key:
        return "Unauthorized"
    if type(brain_key) == str:
        return [brain_key]
    brains = data_adapter.get_brains_list()
    return [brain.name_key for brain in brains]


@mcp.tool()
async def list_brains() -> list[str] | str:
    """
    This tool lists all the brains/memory stores available
    """
    return await asyncio.to_thread(_list_brains_sync)


def _mutate_fact_sync(
    action: str,
    fact_id: str,
    brain_id: str,
    verified_by: str,
    verification_source: str,
    note: str,
) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        record = apply_verification_action(
            graph_adapter,
            fact_id=fact_id,
            brain_id=brain_id,
            action=action,  # type: ignore[arg-type]
            verified_by=verified_by or None,
            verification_source=verification_source or None,
            note=note or None,
        )
        if action == "verify":
            from src.core.saving.conflict_verification import (
                close_open_conflicts_for_verified_fact,
            )

            close_open_conflicts_for_verified_fact(
                data_adapter,
                brain_id,
                fact_id,
                graph=graph_adapter,
                notes=note or None,
            )
        return json.dumps(record.model_dump(mode="json"), default=str)
    except FactNotFound:
        return "Fact not found"
    except Exception as e:
        detail = getattr(e, "detail", None)
        if detail:
            return f"Error: {detail}"
        return f"Error updating fact verification: {e}"


@mcp.tool()
async def verify_fact(
    fact_id: str,
    brain_id: str,
    verified_by: str,
    verification_source: str = "",
    note: str = "",
) -> Any:
    """
    Mark a fact (relationship UUID or fact-representing node UUID) as verified.

    Brain-scoped; requires a valid BrainPAT. verified_by is the user or agent identity.
    """
    return await asyncio.to_thread(
        _mutate_fact_sync,
        "verify",
        fact_id,
        brain_id,
        verified_by,
        verification_source,
        note,
    )


@mcp.tool()
async def reject_fact(
    fact_id: str,
    brain_id: str,
    verified_by: str,
    verification_source: str = "",
    note: str = "",
) -> Any:
    """
    Mark a fact as rejected so retrieve/search/traverse hide it by default.
    """
    return await asyncio.to_thread(
        _mutate_fact_sync,
        "reject",
        fact_id,
        brain_id,
        verified_by,
        verification_source,
        note,
    )


@mcp.tool()
async def reset_fact_verification(
    fact_id: str,
    brain_id: str,
    verified_by: str = "",
    note: str = "",
) -> Any:
    """
    Reset a fact back to unverified. Append-only audit history is retained.
    """
    return await asyncio.to_thread(
        _mutate_fact_sync,
        "reset",
        fact_id,
        brain_id,
        verified_by,
        "",
        note,
    )


def _list_facts_sync(
    brain_id: str,
    state: str,
    include_rejected: bool,
    limit: int,
    skip: int,
) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        normalized = require_state(state or None)
        records, total = list_fact_records(
            graph_adapter,
            brain_id,
            state=normalized,
            include_rejected=include_rejected,
            limit=limit,
            skip=skip,
        )
        return json.dumps(
            {
                "total": total,
                "facts": [record.model_dump(mode="json") for record in records],
            },
            default=str,
        )
    except Exception as e:
        detail = getattr(e, "detail", None)
        if detail:
            return f"Error: {detail}"
        return f"Error listing facts: {e}"


@mcp.tool()
async def list_facts_by_verification(
    brain_id: str,
    state: str = "",
    include_rejected: bool = False,
    limit: int = 50,
    skip: int = 0,
) -> Any:
    """
    List facts in a brain filtered by verification state:
    unverified, verified, or rejected. Default listing hides rejected facts
    unless include_rejected is true or state is rejected.
    """
    return await asyncio.to_thread(
        _list_facts_sync, brain_id, state, include_rejected, limit, skip
    )


def _get_fact_sync(fact_id: str, brain_id: str) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        record = get_fact_record(graph_adapter, fact_id, brain_id)
        return json.dumps(record.model_dump(mode="json"), default=str)
    except FactNotFound:
        return "Fact not found"
    except Exception as e:
        return f"Error reading fact verification: {e}"


@mcp.tool()
async def get_fact_verification(
    fact_id: str,
    brain_id: str,
) -> Any:
    """
    Get the current verification state, verifier info, and append-only audit
    history for a fact.
    """
    return await asyncio.to_thread(_get_fact_sync, fact_id, brain_id)


def _string_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or None
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return parts or None
    return None


def _list_changelogs_sync(
    brain_id: str,
    query_text: str,
    types: Any,
    limit: int,
    skip: int,
    since: str,
    until: str,
) -> Any:
    result = read_changelogs(
        data_adapter,
        brain_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        query_text=query_text or None,
        types=_string_list(types),
        limit=limit,
        skip=skip,
        since=since or None,
        until=until or None,
        graph=graph_adapter,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def list_changelogs(
    brain_id: str,
    query_text: str = "",
    types: list[str] | None = None,
    limit: int = 10,
    skip: int = 0,
    since: str = "",
    until: str = "",
) -> Any:
    """
    List graph changelog entries for a brain, newest first.

    Optional query_text matches the stored change. Optional types restrict the page
    to relationship_created, relationship_deprecated, node_properties_updated,
    or relationship_properties_updated. Optional since/until ISO-8601 datetimes
    return added, superseded, and invalidated facts in that window.
    """
    return await asyncio.to_thread(
        _list_changelogs_sync,
        brain_id,
        query_text,
        types,
        limit,
        skip,
        since,
        until,
    )


def _get_changelog_sync(brain_id: str, changelog_id: str) -> Any:
    result = read_changelog(
        data_adapter,
        brain_id,
        changelog_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def get_changelog(brain_id: str, id: str = "") -> Any:
    """
    Read one graph changelog entry by id.
    """
    return await asyncio.to_thread(_get_changelog_sync, brain_id, id)


def _embed_query(text: str) -> Any:
    return embeddings_adapter.embed_text(text)


def _search_passages(vector: Any, brain_id: str) -> Any:
    return vector_search.search_data(vector, brain_id=brain_id, k=10)


def _list_source_versions_sync(
    brain_id: str,
    source_key: str,
    query: str,
    limit: int,
    skip: int,
) -> Any:
    result = read_source_versions(
        data_adapter,
        brain_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        source_key=source_key,
        query=query,
        limit=limit,
        skip=skip,
        embed=_embed_query,
        search=_search_passages,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def list_source_versions(
    brain_id: str,
    source_key: str = "",
    query: str = "",
    limit: int = 10,
    skip: int = 0,
) -> Any:
    """
    Read the current copy and previous copies of a source.

    Pass source_key when it is known. Otherwise pass a query; passage search
    only discovers the source key, then every stored copy for that key is returned.
    Use skip to page previous copies (source_key) or discovered keys (query).
    """
    return await asyncio.to_thread(
        _list_source_versions_sync,
        brain_id,
        source_key,
        query,
        limit,
        skip,
    )


def _list_conflicts_sync(
    brain_id: str,
    status: str,
    source: str,
    subject: str,
    since: str,
    until: str,
    limit: int,
    skip: int,
) -> Any:
    result = read_conflicts(
        data_adapter,
        brain_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        status=status or None,
        source=source or None,
        subject=subject or None,
        since=since or None,
        until=until or None,
        limit=limit,
        skip=skip,
        graph=graph_adapter,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def list_conflicts(
    brain_id: str,
    status: str = "",
    source: str = "",
    subject: str = "",
    since: str = "",
    until: str = "",
    limit: int = 10,
    skip: int = 0,
) -> Any:
    """
    List cross-source fact conflicts for a brain.
    """
    return await asyncio.to_thread(
        _list_conflicts_sync,
        brain_id,
        status,
        source,
        subject,
        since,
        until,
        limit,
        skip,
    )


def _get_conflict_sync(brain_id: str, conflict_id: str) -> Any:
    result = read_conflict(
        data_adapter,
        brain_id,
        conflict_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        graph=graph_adapter,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def get_conflict(brain_id: str, id: str = "") -> Any:
    """
    Read one conflict with full provenance for both sides.
    """
    return await asyncio.to_thread(_get_conflict_sync, brain_id, id)


def _resolve_conflict_sync(
    brain_id: str,
    conflict_id: str,
    winning_fact_id: str,
    notes: str,
    method: str,
) -> Any:
    result = resolve_conflict_tool(
        data_adapter,
        brain_id,
        conflict_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        winning_fact_id=winning_fact_id or None,
        notes=notes or None,
        method=method or "manual",
        graph=graph_adapter,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def resolve_conflict(
    brain_id: str,
    id: str = "",
    winning_fact_id: str = "",
    notes: str = "",
    method: str = "manual",
) -> Any:
    """
    Resolve a conflict. Pass winning_fact_id to choose a side; method=verified_side
    verifies the winner and rejects the other side via the fact verification API.
    """
    return await asyncio.to_thread(
        _resolve_conflict_sync, brain_id, id, winning_fact_id, notes, method
    )


def _dismiss_conflict_sync(brain_id: str, conflict_id: str, notes: str) -> Any:
    result = dismiss_conflict_tool(
        data_adapter,
        brain_id,
        conflict_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        notes=notes or None,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def dismiss_conflict(brain_id: str, id: str = "", notes: str = "") -> Any:
    """
    Dismiss a conflict without choosing a winning fact.
    """
    return await asyncio.to_thread(_dismiss_conflict_sync, brain_id, id, notes)


def _conflict_report_sync(
    brain_id: str,
    status: str,
    source: str,
    subject: str,
    since: str,
    until: str,
    fmt: str,
    limit: int,
) -> Any:
    result = conflict_report_tool(
        data_adapter,
        brain_id,
        authorized=bool(guard_brainpat(auth_token_var.get(), brain_id)),
        status=status or None,
        source=source or None,
        subject=subject or None,
        since=since or None,
        until=until or None,
        fmt=fmt or "json",
        limit=limit,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


@mcp.tool()
async def conflict_report(
    brain_id: str,
    status: str = "",
    source: str = "",
    subject: str = "",
    since: str = "",
    until: str = "",
    format: str = "json",
    limit: int = 200,
) -> Any:
    """
    Conflict report with summary counts and per-conflict details.
    """
    return await asyncio.to_thread(
        _conflict_report_sync,
        brain_id,
        status,
        source,
        subject,
        since,
        until,
        format,
        limit,
    )
