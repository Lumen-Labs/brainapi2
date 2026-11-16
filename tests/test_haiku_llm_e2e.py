from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.openai_ci_stub import Handler, recorded_calls, reset_calls

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_stub():
    reset_calls()
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2)
            break
        except Exception:
            time.sleep(0.05)
    return server, port


def _stop_stub(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def _json(url: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read() or b"{}")


@pytest.fixture
def anthropic_stub():
    server, port = _start_stub()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        _stop_stub(server)


def test_anthropic_client_ingest_and_retrieve_use_only_haiku(anthropic_stub):
    fake_anthropic = SimpleNamespace(
        api_key="ci-isolated-token",
        base_url=anthropic_stub,
        small_llm_model="claude-haiku-5-5",
        large_llm_model="claude-haiku-5-5",
        openai_compatible_base_url=lambda: f"{anthropic_stub}/v1",
        native_sdk_base_url=lambda: anthropic_stub,
    )
    fake_config = SimpleNamespace(anthropic=fake_anthropic)
    from src.lib.llm.client_anthropic import _AnthropicLLMBase

    with patch("src.lib.llm.client_anthropic.config", fake_config):
        small = _AnthropicLLMBase("claude-haiku-5-5")
        large = _AnthropicLLMBase("claude-haiku-5-5")
        scout_text = small.generate_text(
            "Extract entities as JSON. Ada Lovelace wrote notes."
        )
        architect_json = large.generate_json(
            "Return relationship JSON for: Ada Lovelace wrote notes."
        )
        retrieve_text = small.generate_text(
            "Filter facts for: Who wrote notes? Candidates: 0: Ada wrote notes."
        )
        agent_reply = small.langchain_model.invoke(
            "Scout/Architect agent tool loop for Ada Lovelace wrote notes."
        )

    assert "Ada" in scout_text or "entities" in scout_text
    assert isinstance(architect_json, dict)
    assert retrieve_text
    assert getattr(agent_reply, "content", None) or str(agent_reply)

    calls = _json(f"{anthropic_stub}/llm-calls")
    llm_models = calls["llm_models"]
    assert llm_models, "enrichment and retrieve must call the chat model"
    assert llm_models == ["claude-haiku-5-5"] * len(llm_models)
    kinds = {item["kind"] for item in calls["calls"] if item["kind"] != "embeddings"}
    assert kinds <= {"messages", "chat.completions"}
    assert "messages" in kinds
    assert "chat.completions" in kinds


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
def test_dockerized_anthropic_stub_records_only_haiku():
    image = "python:3.11.14-slim-bookworm"
    name = f"brainapi-haiku-stub-{os.getpid()}"
    port = _free_port()
    subprocess.run(["docker", "pull", image], check=True, capture_output=True)
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:8080",
            "-v",
            f"{ROOT / 'scripts' / 'openai_ci_stub.py'}:/stub.py:ro",
            image,
            "python",
            "/stub.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.4)
        else:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            )
            raise AssertionError(f"stub never became healthy: {logs.stdout} {logs.stderr}")

        _json(f"http://127.0.0.1:{port}/v1/reset", {}, method="POST")
        _json(
            f"http://127.0.0.1:{port}/v1/messages",
            {
                "model": "claude-haiku-5-5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "ingest enrichment"}],
            },
            method="POST",
        )
        _json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": "claude-haiku-5-5",
                "messages": [{"role": "user", "content": "retrieve facts"}],
            },
            method="POST",
        )
        payload = _json(f"http://127.0.0.1:{port}/llm-calls")
        assert payload["llm_models"] == ["claude-haiku-5-5", "claude-haiku-5-5"]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        _ = run


@pytest.mark.skipif(
    not os.getenv("BRAINAPI_E2E_URL") or not os.getenv("HAIKU_STUB_URL"),
    reason="Full docker-stack e2e needs BRAINAPI_E2E_URL and HAIKU_STUB_URL",
)
def test_docker_stack_ingest_and_retrieve_use_only_haiku():
    from scripts.haiku_llm_e2e import run_ingest_and_retrieve

    result = run_ingest_and_retrieve(
        os.environ["BRAINAPI_E2E_URL"],
        os.environ.get("BRAINPAT_TOKEN", "ci-isolated-token"),
        os.environ["HAIKU_STUB_URL"],
    )
    assert result["llm_models"]
    assert set(result["llm_models"]) == {"claude-haiku-5-5"}
