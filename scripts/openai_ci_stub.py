#!/usr/bin/env python3
"""Deterministic OpenAI/Bedrock/Anthropic stub for isolated CI and Haiku routing tests."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

_CALLS: list[dict] = []
_LOCK = threading.Lock()

HAIKU_JSON = json.dumps(
    {
        "entities": [
            {
                "type": "PERSON",
                "name": "Ada",
                "description": "A person mentioned in the text",
                "properties": {},
            }
        ],
        "new_nodes": [],
        "relationships": [],
        "keep_indices": [0],
    }
)


def recorded_calls() -> list[dict]:
    with _LOCK:
        return list(_CALLS)


def reset_calls() -> None:
    with _LOCK:
        _CALLS.clear()


def _record(kind: str, model: str, path: str) -> None:
    with _LOCK:
        _CALLS.append({"kind": kind, "model": model, "path": path})


def _embedding(text: str, dimensions: int = 3072) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _path(handler: BaseHTTPRequestHandler) -> str:
    return urlparse(handler.path).path.rstrip("/") or "/"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        path = _path(self)
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path in {"/llm-calls", "/v1/llm-calls"}:
            calls = recorded_calls()
            self._json(
                200,
                {
                    "calls": calls,
                    "models": [item["model"] for item in calls],
                    "llm_models": [
                        item["model"]
                        for item in calls
                        if item["kind"] in {"messages", "chat.completions", "converse"}
                    ],
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = _path(self)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            request = {}
        if path in {"/reset", "/v1/reset"}:
            reset_calls()
            self._json(200, {"status": "ok"})
            return
        if path.endswith("/embeddings"):
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            dimensions = int(request.get("dimensions") or 3072)
            data = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": _embedding(str(text), dimensions),
                }
                for index, text in enumerate(inputs)
            ]
            model = request.get("model", "ci-embedding")
            _record("embeddings", model, path)
            self._json(
                200,
                {
                    "object": "list",
                    "model": model,
                    "data": data,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
            return
        if path.endswith("/converse") or path.endswith("/invoke"):
            parts = [item for item in path.split("/") if item]
            model = request.get("modelId") or request.get("model") or "us.anthropic.claude-haiku-5-5"
            if "model" in parts:
                index = parts.index("model")
                if index + 1 < len(parts):
                    model = unquote(parts[index + 1])
            _record("converse", model, path)
            self._json(
                200,
                {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": HAIKU_JSON}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "totalTokens": 2,
                    },
                },
            )
            return
        if path.endswith("/messages"):
            model = request.get("model", "claude-haiku-5-5")
            _record("messages", model, path)
            self._json(
                200,
                {
                    "id": "msg_ci",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": HAIKU_JSON}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
            return
        if path.endswith("/chat/completions"):
            model = request.get("model", "claude-haiku-5-5")
            _record("chat.completions", model, path)
            self._json(
                200,
                {
                    "id": "chatcmpl-ci",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": HAIKU_JSON},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
            return
        self._json(404, {"error": {"message": "unsupported CI stub endpoint"}})

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
