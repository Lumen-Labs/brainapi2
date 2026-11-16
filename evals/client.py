from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from evals.config import TERMINAL_TASK_STATUSES, Settings


@dataclass
class TimedResult:
    data: Any
    latency_ms: float
    status_code: int = 200


@dataclass
class BrainAPIClient:
    settings: Settings
    timeout: float = 180.0
    max_retries: int = 5
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.settings.require_brainapi()
        self._client = httpx.Client(
            base_url=self.settings.brainapi_url,
            headers={
                "Authorization": f"Bearer {self.settings.brainpat_token}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BrainAPIClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        allow_http_error: bool = False,
    ) -> TimedResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = self._client.request(
                    method, path, json=json, params=params
                )
                latency_ms = (time.perf_counter() - started) * 1000
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2**attempt * 0.5, 30))
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                if response.status_code >= 400 and not allow_http_error:
                    detail = ""
                    try:
                        detail = response.text[:500]
                    except Exception:
                        detail = ""
                    raise RuntimeError(
                        f"{method} {path} returned {response.status_code}. body={detail!r}"
                    )
                if response.status_code == 204 or not response.content:
                    return TimedResult(
                        data=None,
                        latency_ms=latency_ms,
                        status_code=response.status_code,
                    )
                try:
                    payload = response.json()
                except Exception:
                    payload = {"raw": response.text[:2000]}
                return TimedResult(
                    data=payload,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                time.sleep(min(2**attempt * 0.5, 30))
        raise RuntimeError(f"Request failed after retries: {method} {path}: {last_error}")

    def openapi(self) -> TimedResult:
        return self._request("GET", "/openapi.json", allow_http_error=True)

    def ingest_text(
        self,
        text: str,
        brain_id: str,
        *,
        source_timestamp: str | None = None,
        skip_enrichment: bool = False,
        observate_for: list[str] | None = None,
        preferred_extraction_entities: list[str] | None = None,
        meta_keys: dict[str, Any] | None = None,
    ) -> TimedResult:
        body: dict[str, Any] = {
            "data": {"data_type": "text", "text_data": text},
            "brain_id": brain_id,
            "observate_for": observate_for
            or [
                "rules, clauses, and the dates they were in force",
                "policy statements and their sources",
                "procedures, owners, and dependencies between steps",
            ],
            "preferred_extraction_entities": preferred_extraction_entities
            or ["Clause", "Policy", "Procedure", "Person", "Date", "System"],
            "skip_enrichment": skip_enrichment,
        }
        if source_timestamp:
            body["source_timestamp"] = source_timestamp
        if meta_keys:
            body["meta_keys"] = meta_keys
        return self._request("POST", "/ingest/", json=body)

    def get_task(self, task_id: str, brain_id: str) -> TimedResult:
        return self._request("GET", f"/tasks/{task_id}", params={"brain_id": brain_id})

    def wait_for_task(
        self,
        task_id: str,
        brain_id: str,
        *,
        timeout_s: float = 600.0,
        poll_interval_s: float = 2.0,
    ) -> TimedResult:
        started = time.perf_counter()
        while True:
            result = self.get_task(task_id, brain_id)
            status = (result.data or {}).get("status", "unknown")
            if status in TERMINAL_TASK_STATUSES:
                return TimedResult(
                    data=result.data,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    status_code=result.status_code,
                )
            if time.perf_counter() - started > timeout_s:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {timeout_s}s (last status={status})"
                )
            time.sleep(poll_interval_s)

    def retrieve_context(
        self,
        text: str,
        brain_id: str,
        *,
        extra_body: dict[str, Any] | None = None,
        historical_limit: int = 10,
        max_passages: int = 8,
        max_facts: int = 40,
        apply_fact_filter: bool = True,
        use_ppr: bool = True,
        sufficiency_retry: bool = False,
    ) -> TimedResult:
        body: dict[str, Any] = {
            "text": text,
            "brain_id": brain_id,
            "historical_limit": historical_limit,
            "max_passages": max_passages,
            "max_facts": max_facts,
            "apply_fact_filter": apply_fact_filter,
            "use_ppr": use_ppr,
            "sufficiency_retry": sufficiency_retry,
        }
        if extra_body:
            body.update(extra_body)
        return self._request("POST", "/retrieve/context", json=body)

    def request_path(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        allow_http_error: bool = True,
    ) -> TimedResult:
        return self._request(
            method,
            path,
            json=json,
            params=params,
            allow_http_error=allow_http_error,
        )
