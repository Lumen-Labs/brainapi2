import logging
import os
import re
from hashlib import sha1
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute
from uvicorn import run

from src.constants.data import BRAIN_VERSION
from src.services.api.console_static import SPAStaticFiles
from src.services.api.errors import install_error_handlers
from src.services.api.openapi import install_openapi_contract

from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.middlewares.brains import BrainMiddleware
from src.services.api.middlewares.ingest_publication import IngestPublicationMiddleware
from src.services.api.routes.ingest import ingest_router
from src.services.api.routes.meta import meta_router
from src.services.api.routes.model import model_router
from src.services.api.routes.public import public_router
from src.services.api.routes.retrieve import retrieve_router
from src.services.api.routes.system import system_router
from src.services.api.routes.tasks import tasks_router
from src.lib.tracing.middleware import TraceMiddleware
from src.lib.tracing.runtime import start_runtime_monitoring, stop_runtime_monitoring

logger = logging.getLogger("brainapi.plugins")

PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", str(_project_root / "plugins")))
CONSOLE_DIST = _project_root / "console" / "dist"


def _console_enabled() -> bool:
    return os.getenv("CONSOLE_ENABLED", "true").strip().lower() != "false"


def _production_mode() -> bool:
    return os.getenv("ENV", "production").strip().lower() != "development"


def _cors_allowed_origins() -> list[str]:
    configured = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if not configured and not _production_mode():
        return ["*"]
    if "*" in configured and _production_mode():
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS='*' is allowed only when ENV=development"
        )
    return configured


def _enforce_plugin_results(results: dict[str, bool]) -> None:
    failed = sorted(name for name, loaded in results.items() if not loaded)
    if not failed:
        return
    default_policy = "fail" if _production_mode() else "warn"
    policy = os.getenv("PLUGIN_FAILURE_POLICY", default_policy).strip().lower()
    if policy not in {"fail", "warn"}:
        raise RuntimeError("PLUGIN_FAILURE_POLICY must be 'fail' or 'warn'")
    if policy == "fail":
        raise RuntimeError(f"Required plugins failed to load: {', '.join(failed)}")


def stable_operation_id(route: APIRoute) -> str:
    """Generate function-calling-safe IDs that are stable across schema exports."""

    tag = route.tags[0] if route.tags else "api"
    method = sorted(route.methods or {"get"})[0].lower()
    raw = f"{tag}_{route.name}_{method}".lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if len(normalized) <= 64:
        return normalized
    digest = sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:55].rstrip('_')}_{digest}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.core.plugins.context import PluginContext
    from src.core.plugins.loader import PluginLoader

    start_runtime_monitoring("brainapi-api")
    ctx = PluginContext.from_app(app)
    loader = PluginLoader(plugins_dir=PLUGINS_DIR, context=ctx)
    results = loader.load_all()

    _log_plugin_banner(loader, results)
    _enforce_plugin_results(results)

    for event_name, handlers in ctx._event_handlers.items():
        if event_name == "startup":
            for handler in handlers:
                await handler() if _is_coroutine(handler) else handler()

    try:
        yield
    finally:
        stop_runtime_monitoring("brainapi-api")

    for event_name, handlers in ctx._event_handlers.items():
        if event_name == "shutdown":
            for handler in handlers:
                await handler() if _is_coroutine(handler) else handler()


def _is_coroutine(fn):
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _log_plugin_banner(loader, results: dict[str, bool]):
    loaded = loader.loaded_plugins
    total = len(results)
    ok = sum(1 for v in results.values() if v)
    failed = total - ok

    lines = [
        "",
        "\033[36m ╔══════════════════════════════════════════════════════╗\033[0m",
        "\033[36m ║\033[0m             \033[1;36m⚡  BrainAPI Plugin System  ⚡\033[0m           \033[36m║\033[0m",
        "\033[36m ╠══════════════════════════════════════════════════════╣\033[0m",
    ]

    if total == 0:
        lines.append(
            "\033[36m ║\033[0m  \033[2mNo plugins installed\033[0m                                \033[36m║\033[0m"
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
            lines.append(f"\033[36m ║\033[0m{padded}{' ' * max(pad, 1)}\033[36m║\033[0m")

    lines.append("\033[36m ╠══════════════════════════════════════════════════════╣\033[0m")

    summary_parts = [f"\033[1;32m{ok} loaded\033[0m"]
    if failed:
        summary_parts.append(f"\033[1;31m{failed} failed\033[0m")
    summary_text = f"  {' · '.join(summary_parts)}"
    visible_summary_len = len(f"  {ok} loaded" + (f" · {failed} failed" if failed else ""))
    summary_pad = 54 - visible_summary_len
    lines.append(f"\033[36m ║\033[0m{summary_text}{' ' * max(summary_pad, 1)}\033[36m║\033[0m")

    lines.append("\033[36m ╚══════════════════════════════════════════════════════╝\033[0m")
    lines.append("")

    print("\n".join(lines))


app = FastAPI(
    title="BrainAPI",
    version=BRAIN_VERSION,
    description="Knowledge, memory, search, recommendation, and agent API.",
    debug=os.getenv("ENV") == "development",
    lifespan=lifespan,
    redirect_slashes=False,
    generate_unique_id_function=stable_operation_id,
)
install_error_handlers(app)

app.add_middleware(IngestPublicationMiddleware)
app.add_middleware(BrainPATMiddleware)
app.add_middleware(BrainMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TraceMiddleware, service_name="brainapi-api")

app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(meta_router)
app.include_router(model_router)
app.include_router(system_router)
app.include_router(tasks_router)
app.include_router(public_router)
install_openapi_contract(app)


@app.get("/", include_in_schema=False)
async def root():
    return Response(content="ok", status_code=200)


if _console_enabled():

    @app.get("/console", include_in_schema=False)
    async def console_redirect():
        return RedirectResponse(url="/console/", status_code=307)

    if CONSOLE_DIST.is_dir():
        app.mount(
            "/console",
            SPAStaticFiles(directory=str(CONSOLE_DIST), html=True),
            name="console",
        )
        logger.info("Local console mounted at /console")
    else:
        logger.warning(
            "Console enabled but %s not found — run: make build-console",
            CONSOLE_DIST,
        )
elif os.getenv("CONSOLE_ENABLED", "").strip().lower() == "true":
    logger.warning(
        "CONSOLE_ENABLED=true but console is disabled — set CONSOLE_ENABLED=false to silence",
    )


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
