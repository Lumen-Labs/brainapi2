import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.middlewares.brains import BrainMiddleware
from src.services.api.routes.ingest import ingest_router
from src.services.api.routes.meta import meta_router
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router
from src.services.api.routes.system import system_router
from src.services.api.routes.tasks import tasks_router

logger = logging.getLogger("brainapi.plugins")

PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", str(_project_root / "plugins")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.core.plugins.context import PluginContext
    from src.core.plugins.loader import PluginLoader

    ctx = PluginContext.from_app(app)
    loader = PluginLoader(plugins_dir=PLUGINS_DIR, context=ctx)
    results = loader.load_all()

    _log_plugin_banner(loader, results)

    for event_name, handlers in ctx._event_handlers.items():
        if event_name == "startup":
            for handler in handlers:
                await handler() if _is_coroutine(handler) else handler()

    yield

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
    debug=os.getenv("ENV") == "development",
    lifespan=lifespan,
)

app.add_middleware(BrainPATMiddleware)
app.add_middleware(BrainMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(meta_router)
app.include_router(model_router)
app.include_router(system_router)
app.include_router(tasks_router)


@app.get("/")
async def root():
    return Response(content="ok", status_code=200)


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000, reload=os.getenv("ENV") == "development")
