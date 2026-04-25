import logging
import os
from pathlib import Path

import dotenv
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

_project_root = Path(__file__).resolve().parent.parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from src.lib.tracing.middleware import TraceMiddleware
from src.services.mcp.main import auth_token_var, mcp

PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", str(_project_root / "plugins")))

logger = logging.getLogger("brainapi.plugins")


def _load_mcp_plugins():
    from src.core.plugins.context import PluginContext
    from src.core.plugins.loader import PluginLoader

    ctx = PluginContext.from_mcp(mcp)
    loader = PluginLoader(plugins_dir=PLUGINS_DIR, context=ctx)
    results = loader.load_all()
    _log_plugin_banner(loader, results)


def _log_plugin_banner(loader, results: dict[str, bool]):
    loaded = loader.loaded_plugins
    total = len(results)
    ok = sum(1 for v in results.values() if v)
    failed = total - ok

    lines = [
        "",
        "\033[35m ╔══════════════════════════════════════════════════════╗\033[0m",
        "\033[35m ║\033[0m          \033[1;35m⚡  BrainAPI MCP Plugin System  ⚡\033[0m          \033[35m║\033[0m",
        "\033[35m ╠══════════════════════════════════════════════════════╣\033[0m",
    ]

    if total == 0:
        lines.append(
            "\033[35m ║\033[0m  \033[2mNo plugins installed\033[0m                                \033[35m║\033[0m"
        )
    else:
        for name, success in results.items():
            manifest = loaded.get(name)
            if success and manifest:
                ver = f"v{manifest.version}"
                status = "\033[32m✔ loaded\033[0m"
                label = f"{manifest.name} ({ver})"
            else:
                status = "\033[31m✘ failed\033[0m"
                label = name
            padded = f"  {status}  {label}"
            visible_len = len(f"  ✔ loaded  {label}")
            pad = 54 - visible_len
            lines.append(f"\033[35m ║\033[0m{padded}{' ' * max(pad, 1)}\033[35m║\033[0m")

    lines.append("\033[35m ╠══════════════════════════════════════════════════════╣\033[0m")
    summary_parts = [f"\033[1;32m{ok} loaded\033[0m"]
    if failed:
        summary_parts.append(f"\033[1;31m{failed} failed\033[0m")
    summary_text = f"  {' · '.join(summary_parts)}"
    visible_summary_len = len(f"  {ok} loaded" + (f" · {failed} failed" if failed else ""))
    summary_pad = 54 - visible_summary_len
    lines.append(f"\033[35m ║\033[0m{summary_text}{' ' * max(summary_pad, 1)}\033[35m║\033[0m")
    lines.append("\033[35m ╚══════════════════════════════════════════════════════╝\033[0m")
    lines.append("")

    print("\n".join(lines))


_load_mcp_plugins()

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
    middleware=[
        Middleware(AuthContextMiddleware),
        Middleware(TraceMiddleware, service_name="brainapi-mcp"),
    ],
    lifespan=_mcp_app.router.lifespan_context,
)
