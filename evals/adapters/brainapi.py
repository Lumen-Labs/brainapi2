from __future__ import annotations

import time
from typing import Any, Callable

from evals.adapters.base import QueryResult
from evals.answer import extractive_answer, llm_answer
from evals.client import BrainAPIClient, TimedResult
from evals.config import Settings, PROMPT_BUDGET
from evals.metrics import pair_key


AsOfPlugin = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ConflictPlugin = Callable[[dict[str, Any], dict[str, Any]], list[tuple[str, str]]]


def _schema_properties(openapi: dict[str, Any], name: str) -> set[str]:
    schemas = ((openapi or {}).get("components") or {}).get("schemas") or {}
    schema = schemas.get(name) or {}
    props = schema.get("properties") or {}
    return set(props.keys())


def _paths(openapi: dict[str, Any]) -> set[str]:
    return set((openapi or {}).get("paths") or {})


def default_as_of_plugin(
    body: dict[str, Any],
    question: dict[str, Any],
    *,
    field: str | None,
    schema_fields: set[str],
) -> tuple[dict[str, Any], str]:
    as_of = question.get("as_of") or question.get("date_from")
    if not as_of:
        return body, "n/a"
    if field and field in schema_fields:
        body = dict(body)
        body[field] = as_of
        return body, "api_field"
    for candidate in ("as_of", "asOf", "point_in_time", "valid_at"):
        if candidate in schema_fields:
            body = dict(body)
            body[candidate] = as_of
            return body, "api_schema"
    return body, "unsupported"


def temporal_conflicts_plugin(payload: dict[str, Any], _question: dict[str, Any]) -> list[tuple[str, str]]:
    rows = payload.get("temporal_conflicts") or []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        hubs = [str(h) for h in (row.get("hubs") or []) if h]
        facts = row.get("facts") or []
        ids: list[str] = list(hubs)
        for fact in facts:
            if isinstance(fact, (list, tuple)) and fact:
                ids.append(str(fact[0]))
            elif fact:
                ids.append(str(fact))
        entity = row.get("entity_uuid")
        if entity:
            ids.append(str(entity))
        uniq = []
        for item in ids:
            if item not in uniq:
                uniq.append(item)
        for i, left in enumerate(uniq):
            for right in uniq[i + 1 :]:
                key = pair_key(left, right)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
    return pairs


class BrainAPIAdapter:
    name = "brainapi"

    def __init__(
        self,
        settings: Settings,
        *,
        brain_id: str,
        answerer: str = "extractive",
        skip_enrichment: bool = False,
        as_of_plugin: AsOfPlugin | None = None,
        conflict_plugin: ConflictPlugin | None = None,
        client: BrainAPIClient | None = None,
    ) -> None:
        self.settings = settings
        self.brain_id = brain_id
        self.answerer = answerer
        self.skip_enrichment = skip_enrichment
        self._as_of_plugin = as_of_plugin
        self._conflict_plugin = conflict_plugin or temporal_conflicts_plugin
        self.client = client
        self.ingest_stats: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {
            "as_of": False,
            "as_of_mode": "pending_product",
            "conflict_pairs": False,
            "conflict_mode": "pending_product",
            "pluggable": True,
            "status": "probed_later",
            "context_endpoint": "/retrieve/context",
            "note": (
                "as-of queries and conflict-pair detection are not first-class on "
                "current main. This adapter probes OpenAPI and optional env overrides "
                "(EVAL_AS_OF_FIELD, EVAL_CONFLICT_PATH) and otherwise runs questions "
                "as written against /retrieve/context."
            ),
        }
        self._schema_fields: set[str] = set()
        self._openapi_paths: set[str] = set()
        self._probed = False

    def _ensure_client(self) -> BrainAPIClient:
        if self.client is None:
            self.client = BrainAPIClient(self.settings)
        return self.client

    def probe(self) -> dict[str, Any]:
        client = self._ensure_client()
        result = client.openapi()
        payload = result.data if isinstance(result.data, dict) else {}
        self._schema_fields = _schema_properties(payload, "GetContextRequestBody")
        self._openapi_paths = _paths(payload)
        as_of_fields = {"as_of", "asOf", "point_in_time", "valid_at"}
        env_field = self.settings.as_of_field
        has_as_of = bool((self._schema_fields & as_of_fields) or (env_field and env_field in self._schema_fields))
        conflict_path = self.settings.conflict_path
        has_conflict_path = bool(conflict_path and conflict_path in self._openapi_paths)
        has_temporal = "temporal_conflicts" in (
            ((payload.get("components") or {}).get("schemas") or {})
            .get("GetContextResponse", {})
            .get("properties") or {}
        )
        self.capabilities.update(
            {
                "as_of": has_as_of,
                "as_of_mode": "api_field" if has_as_of else "unsupported_on_this_deployment",
                "conflict_pairs": has_conflict_path,
                "conflict_mode": (
                    "dedicated_endpoint"
                    if has_conflict_path
                    else ("temporal_conflicts_sidecar" if has_temporal else "unsupported_on_this_deployment")
                ),
                "temporal_conflicts_sidecar": has_temporal,
                "openapi_status": result.status_code,
                "status": "ready",
                "schema_as_of_fields": sorted(self._schema_fields & as_of_fields),
            }
        )
        self._probed = True
        return self.capabilities

    def ingest(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        client = self._ensure_client()
        if not self._probed:
            try:
                self.probe()
            except Exception as exc:
                self.capabilities["probe_error"] = str(exc)
        records = []
        started = time.perf_counter()
        for doc in docs:
            header = []
            if doc.get("id"):
                header.append(f"Document id: {doc['id']}")
            if doc.get("title"):
                header.append(f"Title: {doc['title']}")
            if doc.get("clause_id"):
                header.append(f"Clause: {doc['clause_id']}")
            if doc.get("version"):
                header.append(f"Version: {doc['version']}")
            if doc.get("valid_from"):
                header.append(f"Valid from: {doc['valid_from']}")
            if doc.get("valid_to"):
                header.append(f"Valid to: {doc['valid_to']}")
            if doc.get("statement_id"):
                header.append(f"Statement id: {doc['statement_id']}")
            body = "\n".join(header + ["", str(doc.get("text") or "")])
            submitted = client.ingest_text(
                body,
                self.brain_id,
                source_timestamp=str(doc.get("valid_from") or doc.get("source_timestamp") or "")
                or None,
                skip_enrichment=self.skip_enrichment,
            )
            task_id = (submitted.data or {}).get("task_id")
            waited = None
            if task_id:
                waited = client.wait_for_task(str(task_id), self.brain_id)
                status = (waited.data or {}).get("status")
                if status != "completed":
                    raise RuntimeError(
                        f"Ingestion task {task_id} for doc {doc.get('id')} "
                        f"ended with status={status}"
                    )
            records.append(
                {
                    "doc_id": doc.get("id"),
                    "task_id": task_id,
                    "submit_ms": submitted.latency_ms,
                    "wait_ms": waited.latency_ms if waited else None,
                    "status": (waited.data or {}).get("status") if waited else None,
                }
            )
        elapsed = (time.perf_counter() - started) * 1000
        self.ingest_stats = {
            "docs": len(docs),
            "ingest_ms": elapsed,
            "records": records,
            "skip_enrichment": self.skip_enrichment,
            "llm_model": self.settings.llm_model,
            "brain_id": self.brain_id,
            "http_calls": len(docs) * 2,
        }
        return self.ingest_stats

    def query(self, question: dict[str, Any]) -> QueryResult:
        client = self._ensure_client()
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        as_of_mode = "unsupported"
        if self._as_of_plugin:
            extra, as_of_mode = self._as_of_plugin(extra, question)  # type: ignore[misc]
        else:
            extra, as_of_mode = default_as_of_plugin(
                extra,
                question,
                field=self.settings.as_of_field,
                schema_fields=self._schema_fields,
            )
        notes = [f"as_of_mode={as_of_mode}"]
        retrieve = client.retrieve_context(
            str(question.get("question") or ""),
            self.brain_id,
            extra_body=extra or None,
            max_passages=int(PROMPT_BUDGET["max_passages"]),
        )
        payload = retrieve.data if isinstance(retrieve.data, dict) else {}
        passages = self._passages_from_context(payload)
        pairs: list[tuple[str, str]] = []
        http_calls = 1
        if question.get("task") == "conflict_audit":
            path = self.settings.conflict_path
            if path:
                conflict_res = client.request_path(
                    "POST",
                    path,
                    json={
                        "brain_id": self.brain_id,
                        "text": question.get("question"),
                    },
                )
                http_calls += 1
                if conflict_res.status_code < 400 and isinstance(conflict_res.data, dict):
                    pairs = self._pairs_from_payload(conflict_res.data)
                    notes.append("conflict_endpoint")
                    self.capabilities["conflict_pairs"] = True
                    self.capabilities["conflict_mode"] = "dedicated_endpoint"
                else:
                    notes.append(f"conflict_endpoint_http_{conflict_res.status_code}")
            if not pairs:
                pairs = self._conflict_plugin(payload, question)
                if pairs:
                    notes.append("temporal_conflicts_sidecar")
                else:
                    notes.append("conflict_capability_pending")
        answer = self._answer(question, passages, payload)
        e2e_ms = (time.perf_counter() - started) * 1000
        return QueryResult(
            answer=answer["text"],
            passages=passages,
            conflict_pairs=pairs,
            retrieve_ms=retrieve.latency_ms,
            answer_ms=answer["latency_ms"],
            e2e_ms=e2e_ms,
            prompt_tokens=answer["prompt_tokens"],
            completion_tokens=answer["completion_tokens"],
            total_tokens=answer["total_tokens"],
            llm_calls=answer["calls"],
            http_calls=http_calls,
            raw={"context": payload, "as_of_mode": as_of_mode},
            notes=notes,
        )

    def _passages_from_context(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        passages: list[dict[str, Any]] = []
        for index, text in enumerate(payload.get("source_passages") or []):
            passages.append(
                {
                    "id": f"passage-{index}",
                    "doc_id": f"passage-{index}",
                    "title": "source_passage",
                    "text": str(text),
                }
            )
        for index, text in enumerate(payload.get("historical_context") or []):
            passages.append(
                {
                    "id": f"hist-{index}",
                    "doc_id": f"hist-{index}",
                    "title": "historical_context",
                    "text": str(text),
                }
            )
        blob = str(payload.get("text_context") or "").strip()
        if blob and not passages:
            passages.append(
                {
                    "id": "text_context",
                    "doc_id": "text_context",
                    "title": "text_context",
                    "text": blob,
                }
            )
        return passages[: int(PROMPT_BUDGET["max_passages"])]

    def _pairs_from_payload(self, payload: dict[str, Any]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        rows = payload.get("pairs") or payload.get("conflicts") or payload.get("items") or []
        for row in rows:
            if isinstance(row, dict):
                left = str(row.get("left") or row.get("a") or row.get("source") or "")
                right = str(row.get("right") or row.get("b") or row.get("target") or "")
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                left, right = str(row[0]), str(row[1])
            else:
                continue
            if left and right and left != right:
                pairs.append(pair_key(left, right))
        return pairs

    def _answer(
        self,
        question: dict[str, Any],
        passages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.answerer == "llm":
            result = llm_answer(
                self.settings, str(question.get("question") or ""), passages
            )
            return {
                "text": result.text,
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "calls": result.calls,
            }
        result = extractive_answer(question, passages)
        text = result.text
        if not text:
            text = str(payload.get("text_context") or "")[:800]
        return {
            "text": text,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
