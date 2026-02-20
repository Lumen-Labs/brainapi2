from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from mcp.server.streamable_http_manager import (
    StreamableHTTPSessionManager,
    StreamableHTTPASGIApp,
)
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route
from src.services.mcp.main import mcp

transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)

session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    json_response=mcp.settings.json_response,
    stateless=mcp.settings.stateless_http,
    security_settings=transport_security,
)

mcp._session_manager = session_manager

streamable_http_app = StreamableHTTPASGIApp(session_manager)

app = Starlette(
    routes=[
        Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    ],
    lifespan=lambda _: session_manager.run(),
)
