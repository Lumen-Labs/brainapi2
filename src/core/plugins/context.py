from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from src.adapters.cache import CacheAdapter
from src.adapters.data import DataAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.core.plugins.prompts import PromptRegistry, prompt_registry

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI
    from mcp.server import FastMCP
    from src.config import Config


class PluginAdapters:
    def __init__(
        self,
        cache: CacheAdapter,
        graph: GraphAdapter,
        data: DataAdapter,
        vector_store: VectorStoreAdapter,
        llm_small: LLMAdapter,
        llm_large: LLMAdapter,
        embeddings: EmbeddingsAdapter,
        embeddings_small: EmbeddingsAdapter,
    ):
        self.cache = cache
        self.graph = graph
        self.data = data
        self.vector_store = vector_store
        self.llm_small = llm_small
        self.llm_large = llm_large
        self.embeddings = embeddings
        self.embeddings_small = embeddings_small


class PluginContext:
    def __init__(
        self,
        adapters: PluginAdapters,
        prompts: PromptRegistry,
        config: "Config",
        app: Optional["FastAPI"] = None,
        mcp: Optional["FastMCP"] = None,
    ):
        self._app = app
        self.mcp = mcp
        self.adapters = adapters
        self.prompts = prompts
        self.config = config
        self._routers: list[tuple["APIRouter", dict[str, Any]]] = []
        self._event_handlers: dict[str, list[Callable]] = {}

    @classmethod
    def _build_adapters(cls) -> PluginAdapters:
        from src.core.instances import (
            cache_adapter,
            data_adapter,
            embeddings_adapter,
            embeddings_small_adapter,
            graph_adapter,
            llm_large_adapter,
            llm_small_adapter,
            vector_store_adapter,
        )

        return PluginAdapters(
            cache=cache_adapter,
            graph=graph_adapter,
            data=data_adapter,
            vector_store=vector_store_adapter,
            llm_small=llm_small_adapter,
            llm_large=llm_large_adapter,
            embeddings=embeddings_adapter,
            embeddings_small=embeddings_small_adapter,
        )

    @classmethod
    def from_app(cls, app: "FastAPI") -> "PluginContext":
        from src.config import config

        return cls(
            app=app,
            adapters=cls._build_adapters(),
            prompts=prompt_registry,
            config=config,
        )

    @classmethod
    def from_mcp(cls, mcp: "FastMCP") -> "PluginContext":
        from src.config import config

        return cls(
            mcp=mcp,
            adapters=cls._build_adapters(),
            prompts=prompt_registry,
            config=config,
        )

    def include_router(
        self,
        router: "APIRouter",
        skip_pat: bool = False,
        skip_brain: bool = False,
        **kwargs: Any,
    ) -> None:
        if self._app is None:
            return
        self._app.include_router(router, **kwargs)
        self._routers.append((router, kwargs))

        prefix = kwargs.get("prefix", "") or getattr(router, "prefix", "")
        if prefix:
            if skip_pat:
                from src.services.api.middlewares.auth import BrainPATMiddleware
                BrainPATMiddleware.excluded_prefixes.add(prefix)
            if skip_brain:
                from src.services.api.middlewares.brains import BrainMiddleware
                BrainMiddleware.excluded_prefixes.add(prefix)

    def add_middleware(self, middleware_cls: type, **kwargs: Any) -> None:
        if self._app is None:
            return
        self._app.add_middleware(middleware_cls, **kwargs)

    def register_mcp_tool(self, fn: Callable, **kwargs: Any) -> None:
        if self.mcp is None:
            return
        self.mcp.tool(**kwargs)(fn)

    def add_event_handler(self, event: str, handler: Callable) -> None:
        self._event_handlers.setdefault(event, []).append(handler)
