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
from src.lib.neo4j.client import _neo4j_client
from src.services.mcp.oauth_provider import BrainapiMcpOAuthProvider
from src.services.mcp.utils import guard_brainpat
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
<p>Enter your BrainPAT to authorize this MCP client.</p>
<form method="post" action="/mcp-oauth/consent">
<input type="hidden" name="client_id" value="{esc(client_id)}">
<input type="hidden" name="redirect_uri" value="{esc(redirect_uri)}">
<input type="hidden" name="code_challenge" value="{esc(code_challenge)}">
<input type="hidden" name="scope" value="{esc(scope)}">
<input type="hidden" name="resource" value="{esc(resource)}">
<input type="hidden" name="state" value="{esc(state)}">
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
        brainpat = str(form.get("brainpat") or "").strip()
        if not (client_id and redirect_uri and code_challenge and brainpat):
            return HTMLResponse("Missing fields", status_code=400)
        client = await oauth_provider.get_client(client_id)
        if not client:
            return HTMLResponse("Unknown client", status_code=400)
        if guard_brainpat(brainpat) is False:
            return HTMLResponse("Invalid BrainPAT", status_code=401)
        scopes = scope_str.split() if scope_str.strip() else list(_oauth_scopes)
        for scope_value in scopes:
            if scope_value not in _oauth_scopes:
                return HTMLResponse("Invalid scope", status_code=400)
        try:
            ru = AnyUrl(redirect_uri)
            ru = client.validate_redirect_uri(ru)
        except Exception:
            return HTMLResponse("Invalid redirect_uri", status_code=400)
        code = oauth_provider.issue_auth_code(
            client_id=client_id,
            redirect_uri=ru,
            code_challenge=code_challenge,
            scopes=scopes,
            resource=resource,
            state=state,
            brainpat=brainpat,
        )
        redirect_url = oauth_provider.redirect_after_consent(
            redirect_uri=redirect_uri, code=code, state=state
        )
        return RedirectResponse(redirect_url, status_code=302)

    mcp.custom_route("/mcp-oauth/consent", methods=["GET", "POST"])(_mcp_oauth_consent)


@mcp.tool()
def get_search_operation_instructions(message: str) -> str:
    """
    This tool will provide instructions on how to use the search_memory tool.
    """
    return f"""
    The brains are a storage for information and memories, they are powered by multiple dbs.
    The `search_memory` tool will execute graph operations to search the knowledge graph that contains informations.
    {_neo4j_client.graphdb_description}.
    The input must be a JSON object with the following fields:
    - db_query: str: the operation to execute on the graph.
    - brain_id: str: the brain to search in.
    """


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
    This tool will search into the knowledge graph.

    Input must be a JSON object with the following fields:
    - db_query: str: the operation to execute on the graph.
    - brain_id: str: the brain to search in.
    """
    return await asyncio.to_thread(_search_memory_sync, db_query, brain_id)


def _search_semantically_sync(query: str, brain_id: str) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        query_embedding = embeddings_adapter.embed_text(query)
        data_vectors = vector_search.search_nodes(
            query_embedding.embeddings,
            brain_id=brain_id,
            k=5,
        )
        triplets = graph_adapter.get_event_centric_neighbors(
            [v.metadata.get("uuid") for v in data_vectors], brain_id=brain_id
        )
        return triplets
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
async def search_semantically(query: str, brain_id: str) -> Any:
    """
    Search information semantically, given a query and a brain to search in,
    this tool will return a list of graph nodes that are semantically related to the query.

    This tool is useful to search into the graph for information without knowing names or labels.

    Input must be a JSON object with the following fields:
    - query: str: the query to search for.
    - brain_id: str: the brain to search in.
    """
    return await asyncio.to_thread(_search_semantically_sync, query, brain_id)


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
