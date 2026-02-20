from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from contextlib import asynccontextmanager
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send
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


class MCPEndpoint:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)


@asynccontextmanager
async def lifespan(app: Starlette):
    async with session_manager.run():
        yield


mcp_handler = MCPEndpoint()

app = Starlette(
    routes=[
        Route("/mcp", app=mcp_handler, methods=["GET", "POST", "DELETE"]),
    ],
    lifespan=lifespan,
)
