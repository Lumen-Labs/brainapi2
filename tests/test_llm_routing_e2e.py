from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.openai_ci_stub import Handler, recorded_calls, reset_calls

ROOT = Path(__file__).resolve().parents[1]
TEST_MODEL = "test-model-a"


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
def llm_stub():
    server, port = _start_stub()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        _stop_stub(server)


def test_bedrock_client_uses_configured_model(llm_stub):
    class FakeBedrock:
        region = "us-east-1"
        access_key_id = "ci-isolated-token"
        secret_access_key = "ci-isolated-token"
        session_token = None
        endpoint_url = llm_stub
        small_llm_model = TEST_MODEL
        large_llm_model = TEST_MODEL

        def session_kwargs(self):
            return {
                "region_name": self.region,
                "aws_access_key_id": self.access_key_id,
                "aws_secret_access_key": self.secret_access_key,
            }

        def runtime_client_kwargs(self):
            return {"endpoint_url": self.endpoint_url}

    fake_config = SimpleNamespace(bedrock=FakeBedrock())
    from src.lib.llm.client_bedrock import _BedrockLLMBase

    with patch("src.lib.llm.client_bedrock.config", fake_config):
        small = _BedrockLLMBase(TEST_MODEL)
        large = _BedrockLLMBase(TEST_MODEL)
        scout_text = small.generate_text(
            "Extract entities as JSON. Ada Lovelace wrote notes."
        )
        architect_json = large.generate_json(
            "Return relationship JSON for: Ada Lovelace wrote notes."
        )
        retrieve_text = small.generate_text(
            "Filter facts for: Who wrote notes? Candidates: 0: Ada wrote notes."
        )

    assert "Ada" in scout_text or "entities" in scout_text
    assert isinstance(architect_json, dict)
    assert retrieve_text

    calls = _json(f"{llm_stub}/llm-calls")
    llm_models = calls["llm_models"]
    assert llm_models, "enrichment and retrieve must call the chat model"
    assert llm_models == [TEST_MODEL] * len(llm_models)
    kinds = {item["kind"] for item in calls["calls"] if item["kind"] != "embeddings"}
    assert kinds == {"converse"}


def test_openai_client_uses_configured_model(llm_stub):
    fake_openai = SimpleNamespace(
        api_key="ci-isolated-token",
        base_url=f"{llm_stub}/v1",
        small_llm_model=TEST_MODEL,
        large_llm_model=TEST_MODEL,
    )
    fake_config = SimpleNamespace(openai=fake_openai)
    from src.lib.llm.client_openai import _OpenAILLMBase

    with patch("src.lib.llm.client_openai.config", fake_config):
        small = _OpenAILLMBase(TEST_MODEL)
        large = _OpenAILLMBase(TEST_MODEL)
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

    calls = _json(f"{llm_stub}/llm-calls")
    llm_models = calls["llm_models"]
    assert llm_models, "enrichment and retrieve must call the chat model"
    assert llm_models == [TEST_MODEL] * len(llm_models)
    kinds = {item["kind"] for item in calls["calls"] if item["kind"] != "embeddings"}
    assert kinds == {"chat.completions"}


def test_anthropic_client_uses_configured_model(llm_stub):
    fake_anthropic = SimpleNamespace(
        api_key="ci-isolated-token",
        base_url=llm_stub,
        small_llm_model=TEST_MODEL,
        large_llm_model=TEST_MODEL,
        openai_compatible_base_url=lambda: f"{llm_stub}/v1",
        native_sdk_base_url=lambda: llm_stub,
    )
    fake_config = SimpleNamespace(anthropic=fake_anthropic)
    from src.lib.llm.client_anthropic import _AnthropicLLMBase

    with patch("src.lib.llm.client_anthropic.config", fake_config):
        small = _AnthropicLLMBase(TEST_MODEL)
        large = _AnthropicLLMBase(TEST_MODEL)
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

    calls = _json(f"{llm_stub}/llm-calls")
    llm_models = calls["llm_models"]
    assert llm_models, "enrichment and retrieve must call the chat model"
    assert llm_models == [TEST_MODEL] * len(llm_models)
    kinds = {item["kind"] for item in calls["calls"] if item["kind"] != "embeddings"}
    assert kinds <= {"messages", "chat.completions"}
    assert "messages" in kinds
    assert "chat.completions" in kinds


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
def test_dockerized_stub_records_requested_model():
    image = "python:3.11.14-slim-bookworm"
    name = f"brainapi-llm-stub-{os.getpid()}"
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
            f"http://127.0.0.1:{port}/model/{TEST_MODEL}/converse",
            {
                "messages": [
                    {"role": "user", "content": [{"text": "ingest enrichment"}]}
                ]
            },
            method="POST",
        )
        _json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "retrieve facts"}],
            },
            method="POST",
        )
        payload = _json(f"http://127.0.0.1:{port}/llm-calls")
        assert payload["llm_models"] == [TEST_MODEL, TEST_MODEL]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        _ = run


@pytest.mark.skipif(
    not os.getenv("BRAINAPI_E2E_URL") or not os.getenv("LLM_STUB_URL"),
    reason="Full docker-stack e2e needs BRAINAPI_E2E_URL and LLM_STUB_URL",
)
def test_docker_stack_ingest_and_retrieve_use_configured_model():
    from scripts.llm_routing_e2e import run_ingest_and_retrieve

    expected = os.environ.get("LLM_EXPECTED_MODEL", "ci-test-model")
    result = run_ingest_and_retrieve(
        os.environ["BRAINAPI_E2E_URL"],
        os.environ.get("BRAINPAT_TOKEN", "ci-isolated-token"),
        os.environ["LLM_STUB_URL"],
        expected_model=expected,
    )
    assert result["llm_models"]
    assert all(model == expected for model in result["llm_models"])
