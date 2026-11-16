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
