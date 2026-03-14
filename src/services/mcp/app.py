from pathlib import Path

import dotenv
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from src.services.mcp.main import auth_token_var, mcp

_mcp_app = mcp.streamable_http_app()


class AuthContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            token = None
            brainpat = headers.get(b"brainpat")
            if brainpat:
                token = brainpat.decode()
            else:
                raw = (headers.get(b"authorization") or b"").decode()
                if raw.startswith("Bearer: "):
                    token = raw.removeprefix("Bearer: ").strip() or None
                elif raw.startswith("Bearer "):
                    token = raw.removeprefix("Bearer ").strip() or None
            auth_token_var.set(token)
        await self.app(scope, receive, send)


async def _health(_request):
    return JSONResponse({"status": "ok"}, status_code=200)


async def _mcp_info(_request):
    return JSONResponse(
        {"service": "brainapi-mcp", "streamable_http": True, "path": "/mcp"},
        status_code=200,
    )


_custom_routes = [
    Route("/", _health, methods=["GET"]),
    Route("/mcp", _mcp_info, methods=["GET"]),
    Route("/mcp/info", _mcp_info, methods=["GET"]),
]
app = Starlette(
    routes=_custom_routes + list(_mcp_app.routes),
    middleware=[Middleware(AuthContextMiddleware)],
    lifespan=_mcp_app.router.lifespan_context,
)
