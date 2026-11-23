from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import httpx

from evals.client import BrainAPIClient
from evals.config import Settings


class ClientRetryTests(unittest.TestCase):
    def _client(self) -> BrainAPIClient:
        settings = Settings.load()
        settings.brainpat_token = "token"
        client = BrainAPIClient(settings, max_retries=3)
        client._client.close()
        client._client = MagicMock()
        return client

    def test_post_does_not_retry_read_timeout(self) -> None:
        client = self._client()
        client._client.request.side_effect = httpx.ReadTimeout("timed out")
        with self.assertRaises(RuntimeError) as ctx:
            client._request("POST", "/ingest/")
        self.assertIn("not retried", str(ctx.exception))
        self.assertEqual(client._client.request.call_count, 1)

    def test_post_retries_connect_error(self) -> None:
        client = self._client()
        ok = MagicMock()
        ok.status_code = 200
        ok.content = b'{"ok": true}'
        ok.json.return_value = {"ok": True}
        client._client.request.side_effect = [httpx.ConnectError("offline"), ok]
        result = client._request("POST", "/ingest/")
        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(client._client.request.call_count, 2)

    def test_conflict_scan_posts_scan_path_not_list_path(self) -> None:
        client = self._client()
        ok = MagicMock()
        ok.status_code = 202
        ok.content = b'{"task_id": "scan-1"}'
        ok.json.return_value = {"task_id": "scan-1"}
        client._client.request.return_value = ok
        result = client.start_conflict_scan("evalucpolicyconflicts")
        self.assertEqual(result.status_code, 202)
        method, path = client._client.request.call_args.args[:2]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/retrieve/conflicts/scan")
        self.assertEqual(
            client._client.request.call_args.kwargs["json"], {"use_llm": False}
        )
        self.assertEqual(
            client._client.request.call_args.kwargs["params"]["brain_id"],
            "evalucpolicyconflicts",
        )

    def test_list_conflicts_is_get(self) -> None:
        client = self._client()
        ok = MagicMock()
        ok.status_code = 200
        ok.content = b'{"conflicts": [], "count": 0, "total": 0}'
        ok.json.return_value = {"conflicts": [], "count": 0, "total": 0}
        client._client.request.return_value = ok
        result = client.list_conflicts("evalucpolicyconflicts")
        self.assertEqual(result.data["total"], 0)
        method, path = client._client.request.call_args.args[:2]
        self.assertEqual(method, "GET")
        self.assertEqual(path, "/retrieve/conflicts")
        params = client._client.request.call_args.kwargs["params"]
        self.assertEqual(params["brain_id"], "evalucpolicyconflicts")
        self.assertEqual(params["status"], "open")

    def test_get_retries_read_timeout(self) -> None:
        client = self._client()
        ok = MagicMock()
        ok.status_code = 200
        ok.content = b'{"ok": true}'
        ok.json.return_value = {"ok": True}
        client._client.request.side_effect = [httpx.ReadTimeout("timed out"), ok]
        result = client._request("GET", "/tasks/1")
        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(client._client.request.call_count, 2)
