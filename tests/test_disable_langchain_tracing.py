import os
from pathlib import Path

import langsmith

from src.core.agents.core.agent_base import AgentBase
from src.lib.disable_langchain_tracing import disable_langchain_tracing

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINTS = (
    "src/services/api/app.py",
    "src/workers/app.py",
    "src/services/mcp/app.py",
)
FORMER_TRACE_SITES = (
    "src/core/agents/core/agent_base.py",
    "src/core/agents/core/invoke_loop.py",
    "src/core/saving/auto_kg.py",
    "src/workers/tasks/ingestion.py",
)


def test_disable_langchain_tracing_forces_false(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    disable_langchain_tracing()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_TRACING"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_agent_base_invoke_does_not_use_langsmith_tracing(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("langsmith.tracing_context should not be called")

    monkeypatch.setattr(langsmith, "tracing_context", boom)
    monkeypatch.setattr(
        "src.core.agents.core.agent_base.run_invoke_loop",
        lambda agent, messages, config: {"messages": messages, "config": config},
    )
    agent = AgentBase.__new__(AgentBase)
    result = agent.invoke({"messages": ["hi"]}, config={"tags": ["agent_base"]})
    assert result == {"messages": ["hi"], "config": {"tags": ["agent_base"]}}


def test_entrypoints_import_and_call_disable_helper():
    for relative_path in ENTRYPOINTS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert (
            "from src.lib.disable_langchain_tracing import disable_langchain_tracing"
            in source
        )
        assert "disable_langchain_tracing()" in source


def test_former_trace_sites_do_not_use_langsmith():
    for relative_path in FORMER_TRACE_SITES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "tracing_context" not in source
        assert "wait_for_all_tracers" not in source
        assert "LANGSMITH" not in source
        assert "langsmith" not in source
