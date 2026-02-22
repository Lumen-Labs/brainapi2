from pathlib import Path

import dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from src.services.mcp.main import mcp

_mcp_app = mcp.streamable_http_app()


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
    lifespan=_mcp_app.router.lifespan_context,
)
