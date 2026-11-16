#!/usr/bin/env python3
"""Exercise ingest-with-enrichment and retrieve against a mocked LLM stub."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_EXPECTED_MODEL = "ci-test-model"


def _request(
    base: str,
    method: str,
    path: str,
    *,
    token: str,
    brain_id: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if brain_id:
        headers["X-Brain-ID"] = brain_id
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}", data=payload, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    try:
        parsed = json.loads(raw or b"null")
    except json.JSONDecodeError:
        parsed = raw.decode("utf-8", errors="replace")
    return status, parsed


def _wait_task(base: str, token: str, brain_id: str, task_id: str) -> dict[str, Any]:
    deadline = time.time() + 180
    while time.time() < deadline:
        status, body = _request(
            base, "GET", f"/tasks/{task_id}", token=token, brain_id=brain_id
        )
        if status == 200 and isinstance(body, dict):
            state = (body.get("status") or body.get("state") or "").lower()
            if state in {"completed", "failed", "partial_failed"}:
                return body
        time.sleep(1.5)
    raise TimeoutError(f"ingestion task {task_id} did not finish")


def run_ingest_and_retrieve(
    api_url: str,
    token: str,
    stub_url: str,
    expected_model: str = DEFAULT_EXPECTED_MODEL,
) -> dict[str, Any]:
    expected = (expected_model or DEFAULT_EXPECTED_MODEL).strip()
    brain_id = f"llme2e{int(time.time())}"
    reset_status, _ = _request(stub_url, "POST", "/v1/reset", token=token, body={})
    if reset_status >= 400:
        _request(stub_url, "POST", "/reset", token=token, body={})

    create_status, brain = _request(
        api_url,
        "POST",
        "/system/brains",
        token=token,
        body={"brain_id": brain_id},
    )
    if create_status != 200 or not isinstance(brain, dict) or not brain.get("pat"):
        raise RuntimeError(f"brain create failed: {create_status} {brain}")
    brain_token = str(brain["pat"])

    status, ingested = _request(
        api_url,
        "POST",
        "/ingest/",
        token=brain_token,
        brain_id=brain_id,
        body={
            "brain_id": brain_id,
            "data": {
                "data_type": "text",
                "text_data": "Ada Lovelace wrote the first computer notes in 1843.",
            },
            "skip_enrichment": False,
        },
    )
    if status not in {200, 202} or not isinstance(ingested, dict):
        raise RuntimeError(f"ingest failed: {status} {ingested}")
    task_id = ingested.get("task_id")
    if not task_id:
        raise RuntimeError(f"ingest response missing task_id: {ingested}")
    task = _wait_task(api_url, brain_token, brain_id, str(task_id))
    task_state = (task.get("status") or task.get("state") or "").lower()
    if task_state != "completed":
        raise RuntimeError(f"ingest task ended as {task_state}: {task}")

    retrieve_status, retrieved = _request(
        api_url,
        "POST",
        "/retrieve/context",
        token=brain_token,
        brain_id=brain_id,
        body={
            "text": "Who wrote computer notes?",
            "brain_id": brain_id,
            "apply_fact_filter": False,
        },
    )
    if retrieve_status != 200:
        raise RuntimeError(f"retrieve failed: {retrieve_status} {retrieved}")

    _, calls = _request(stub_url, "GET", "/llm-calls", token=token)
    llm_models = []
    if isinstance(calls, dict):
        llm_models = list(calls.get("llm_models") or [])
    unexpected = sorted(
        {model for model in llm_models if (model or "").strip() != expected}
    )
    if unexpected:
        raise RuntimeError(
            f"chat models other than {expected!r} were used: {unexpected}; all calls={calls}"
        )
    if not llm_models:
        raise RuntimeError(
            "enrichment produced no chat-model calls on the mocked LLM stub"
        )
    return {
        "brain_id": brain_id,
        "task_id": task_id,
        "retrieve_status": retrieve_status,
        "expected_model": expected,
        "llm_models": llm_models,
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stub-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="")
    parser.add_argument("--expected-model", default=DEFAULT_EXPECTED_MODEL)
    args = parser.parse_args()
    token = args.token or os.environ.get("BRAINPAT_TOKEN", "")
    result = run_ingest_and_retrieve(
        args.api_url,
        token,
        args.stub_url,
        expected_model=args.expected_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
