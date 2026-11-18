from __future__ import annotations

import re
import time
from typing import Any, Callable

from evals.adapters.base import QueryResult
from evals.answer import extractive_answer, llm_answer
from evals.client import BrainAPIClient
from evals.config import Settings, PROMPT_BUDGET
from evals.metrics import pair_key


AsOfPlugin = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ConflictPlugin = Callable[[dict[str, Any], dict[str, Any]], list[tuple[str, str]]]

AS_OF_FIELDS = ("as_of", "asOf", "point_in_time", "valid_at")
RANGE_END_FIELDS = ("date_to", "until", "valid_to", "as_of_end", "end_date")
AS_OF_TASKS = frozenset({"as_of", "changelog"})
_REMAP_NOTES = frozenset(
    {
        "conflict_ids_mapped",
        "conflict_mapping_unsupported",
        "conflict_producer_empty",
        "conflict_capability_pending",
    }
)
_STATEMENT_ID_RE = re.compile(r"Statement id:\s*([^\n]+)", re.I)
_DOCUMENT_ID_RE = re.compile(r"Document id:\s*([^\n]+)", re.I)


def _schema_properties(openapi: dict[str, Any], name: str) -> set[str]:
    schemas = ((openapi or {}).get("components") or {}).get("schemas") or {}
    schema = schemas.get(name) or {}
    props = schema.get("properties") or {}
    return set(props.keys())


def _paths(openapi: dict[str, Any]) -> set[str]:
    return set((openapi or {}).get("paths") or {})


def gold_id_for_doc(doc: dict[str, Any]) -> str:
    return str(doc.get("statement_id") or doc.get("id") or "").strip()


def ingest_meta_keys(doc: dict[str, Any]) -> dict[str, str]:
    keys: dict[str, str] = {}
    doc_id = str(doc.get("id") or "").strip()
    statement_id = str(doc.get("statement_id") or "").strip()
    clause_id = str(doc.get("clause_id") or "").strip()
    if doc_id:
        keys["eval_doc_id"] = doc_id
    if statement_id:
        keys["statement_id"] = statement_id
    if clause_id:
        keys["clause_id"] = clause_id
    source_key = clause_id or statement_id or doc_id
    if source_key:
        keys["source_key"] = source_key
    return keys


def parse_eval_ids_from_text(text: str) -> dict[str, str]:
    blob = str(text or "")
    statement = ""
    document = ""
    match = _STATEMENT_ID_RE.search(blob)
    if match:
        statement = match.group(1).strip()
    match = _DOCUMENT_ID_RE.search(blob)
    if match:
        document = match.group(1).strip()
    out: dict[str, str] = {}
    if statement:
        out["statement_id"] = statement
    if document:
        out["doc_id"] = document
    return out


def _needs_as_of(question: dict[str, Any]) -> bool:
    if question.get("as_of") or question.get("date_from") or question.get("date_to"):
        return True
    return str(question.get("task") or "") in AS_OF_TASKS


def _is_range_question(question: dict[str, Any]) -> bool:
    if question.get("date_to"):
        return True
    return str(question.get("task") or "") == "changelog"


def _apply_range_bounds(
    body: dict[str, Any],
    question: dict[str, Any],
    schema_fields: set[str],
) -> dict[str, Any]:
    date_to = question.get("date_to")
    if not date_to:
        if "include_history" in schema_fields:
            body["include_history"] = True
        return body
    for candidate in RANGE_END_FIELDS:
        if candidate in schema_fields:
            body[candidate] = date_to
            break
    if "include_history" in schema_fields:
        body["include_history"] = True
    return body


def default_as_of_plugin(
    body: dict[str, Any],
    question: dict[str, Any],
    *,
    field: str | None,
    schema_fields: set[str],
) -> tuple[dict[str, Any], str]:
    as_of = question.get("as_of") or question.get("date_from")
    if not _needs_as_of(question):
        return body, "n/a"
    if _is_range_question(question):
        body = dict(body)
        ranged = _apply_range_bounds(body, question, schema_fields)
        if any(candidate in ranged for candidate in RANGE_END_FIELDS):
            return ranged, "api_schema"
        if ranged.get("include_history"):
            return ranged, "include_history"
        return ranged, "unsupported"
    if field and field in schema_fields and as_of:
        body = dict(body)
        body[field] = as_of
        return body, "api_field"
    if as_of:
        for candidate in AS_OF_FIELDS:
            if candidate in schema_fields:
                body = dict(body)
                body[candidate] = as_of
                return body, "api_schema"
    if "include_history" in schema_fields:
        body = dict(body)
        body["include_history"] = True
        return body, "include_history"
    if as_of:
        return body, "unsupported"
    return body, "n/a"


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


def _node_parts(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return {
            "uuid": str(node.get("uuid") or node.get("id") or "").strip(),
            "name": str(node.get("name") or "").strip(),
            "properties": node.get("properties") if isinstance(node.get("properties"), dict) else {},
            "metadata": node.get("metadata") if isinstance(node.get("metadata"), dict) else {},
        }
    return {
        "uuid": str(getattr(node, "uuid", "") or getattr(node, "id", "") or "").strip(),
        "name": str(getattr(node, "name", "") or "").strip(),
        "properties": getattr(node, "properties", None) or {},
        "metadata": getattr(node, "metadata", None) or {},
    }


def _has_pair_collection(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(isinstance(payload.get(key), (list, tuple)) for key in ("pairs", "conflicts", "items"))


def remap_conflict_pairs(
    pairs: list[tuple[str, str]],
    id_map: dict[str, str],
) -> dict[str, Any]:
    mapped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    unmapped: list[tuple[str, str]] = []
    for left, right in pairs:
        mapped_left = id_map.get(str(left))
        mapped_right = id_map.get(str(right))
        if mapped_left and mapped_right and mapped_left != mapped_right:
            key = pair_key(mapped_left, mapped_right)
            if key not in seen:
                seen.add(key)
                mapped.append(key)
        else:
            unmapped.append(pair_key(str(left), str(right)))
    supported = not unmapped
    return {
        "pairs": mapped,
        "unmapped": unmapped,
        "supported": supported,
        "n_raw": len(pairs),
        "n_mapped": len(mapped),
        "n_unmapped": len(unmapped),
    }


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
                "as-of queries are feature-detected on GetContextRequestBody. "
                "Until a true as-of field lands, versioned-rule questions send "
                "include_history=true when that flag exists (not a date filter). "
                "A native as-of field takes precedence when OpenAPI exposes it. "
                "Conflict pairs from "
                "temporal_conflicts are mapped back to ingested statement/doc ids "
                "before scoring; unmapped graph UUIDs are reported unsupported."
            ),
        }
        self._schema_fields: set[str] = set()
        self._openapi_paths: set[str] = set()
        self._probed = False
        self._id_map: dict[str, str] = {}
        self._gold_ids: set[str] = set()
        self._locked: set[str] = set()
        self._ambiguous: set[str] = set()
        self._id_candidates: dict[str, set[str]] = {}
        self._conflict_outcomes: list[str] = []
        self._pending_conflict_jobs: list[dict[str, Any]] = []
        self._retrieve_payloads: list[dict[str, Any]] = []

    def _ensure_client(self) -> BrainAPIClient:
        if self.client is None:
            self.client = BrainAPIClient(self.settings)
        return self.client

    def _remember(self, alias: str | None, canonical: str | None, *, lock: bool = False) -> None:
        gold = str(canonical or "").strip()
        key = str(alias or "").strip()
        if not gold:
            return
        self._gold_ids.add(gold)
        self._id_map[gold] = gold
        if not key or key == gold:
            if lock and key:
                self._locked.add(key)
            return
        if key in self._ambiguous:
            return
        if key in self._locked and self._id_map.get(key) != gold:
            return
        candidates = self._id_candidates.setdefault(key, set())
        candidates.add(gold)
        if len(candidates) > 1:
            self._id_map.pop(key, None)
            self._ambiguous.add(key)
            return
        self._id_map[key] = gold
        if lock:
            self._locked.add(key)

    def _canonical(self, raw: str | None) -> str | None:
        value = str(raw or "").strip()
        if not value or value in self._ambiguous:
            return None
        if value in self._id_map:
            return self._id_map[value]
        if value in self._gold_ids:
            return value
        return None

    def _remember_text(self, text: str | None) -> dict[str, str]:
        parsed = parse_eval_ids_from_text(text or "")
        statement = parsed.get("statement_id")
        doc_id = parsed.get("doc_id")
        canonical = self._canonical(statement) or self._canonical(doc_id)
        if canonical:
            self._remember(statement, canonical)
            self._remember(doc_id, canonical)
        return parsed

    def _attribution_candidates(self, row: dict[str, Any]) -> list[Any]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
        return [
            row.get("statement_id"),
            metadata.get("statement_id") if isinstance(metadata, dict) else None,
            properties.get("statement_id") if isinstance(properties, dict) else None,
            row.get("eval_doc_id"),
            metadata.get("eval_doc_id") if isinstance(metadata, dict) else None,
            properties.get("eval_doc_id") if isinstance(properties, dict) else None,
            row.get("doc_id"),
            row.get("document_id"),
            metadata.get("doc_id") if isinstance(metadata, dict) else None,
            metadata.get("document_id") if isinstance(metadata, dict) else None,
            properties.get("doc_id") if isinstance(properties, dict) else None,
            properties.get("document_id") if isinstance(properties, dict) else None,
        ]

    def _remember_metadata(self, row: dict[str, Any] | None) -> str | None:
        if not isinstance(row, dict):
            return None
        candidates = self._attribution_candidates(row)
        canonical = None
        for item in candidates:
            canonical = self._canonical(str(item) if item is not None else None)
            if canonical:
                break
        aliases = [
            row.get("id"),
            row.get("uuid"),
            row.get("chunk_id"),
            row.get("source_chunk_id"),
            *(row.get("source_chunk_ids") or []),
        ]
        if not canonical:
            for alias in aliases:
                canonical = self._canonical(str(alias) if alias is not None else None)
                if canonical:
                    break
        parsed = self._remember_text(str(row.get("text") or row.get("body") or row.get("content") or ""))
        if not canonical:
            canonical = self._canonical(parsed.get("statement_id")) or self._canonical(parsed.get("doc_id"))
        if canonical:
            for alias in aliases:
                self._remember(str(alias) if alias is not None else None, canonical)
            for item in candidates:
                text = str(item or "").strip()
                if text:
                    self._remember(text, canonical)
        return canonical

    def probe(self) -> dict[str, Any]:
        client = self._ensure_client()
        result = client.openapi()
        payload = result.data if isinstance(result.data, dict) else {}
        self._schema_fields = _schema_properties(payload, "GetContextRequestBody")
        self._openapi_paths = _paths(payload)
        env_field = self.settings.as_of_field
        has_env_as_of = bool(env_field and env_field in self._schema_fields)
        has_schema_as_of = bool(self._schema_fields & set(AS_OF_FIELDS))
        has_include_history = "include_history" in self._schema_fields
        if has_env_as_of:
            as_of_mode = "api_field"
            has_as_of = True
        elif has_schema_as_of:
            as_of_mode = "api_schema"
            has_as_of = True
        elif has_include_history:
            as_of_mode = "include_history"
            has_as_of = True
        else:
            as_of_mode = "unsupported_on_this_deployment"
            has_as_of = False
        conflict_path = self.settings.conflict_path
        has_conflict_path = bool(conflict_path and conflict_path in self._openapi_paths)
        has_temporal = "temporal_conflicts" in (
            ((payload.get("components") or {}).get("schemas") or {})
            .get("GetContextResponse", {})
            .get("properties") or {}
        )
        if has_conflict_path:
            conflict_mode = "dedicated_endpoint"
            conflict_pairs = True
        elif has_temporal:
            conflict_mode = "temporal_conflicts_sidecar"
            conflict_pairs = False
        else:
            conflict_mode = "unsupported_on_this_deployment"
            conflict_pairs = False
        self.capabilities.update(
            {
                "as_of": has_as_of,
                "as_of_mode": as_of_mode,
                "include_history": has_include_history,
                "conflict_pairs": conflict_pairs,
                "conflict_mode": conflict_mode,
                "temporal_conflicts_sidecar": has_temporal,
                "openapi_status": result.status_code,
                "status": "ready",
                "schema_as_of_fields": sorted(self._schema_fields & set(AS_OF_FIELDS)),
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
            canonical = gold_id_for_doc(doc)
            self._remember(str(doc.get("id") or ""), canonical, lock=True)
            self._remember(str(doc.get("statement_id") or ""), canonical, lock=True)
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
                meta_keys=ingest_meta_keys(doc) or None,
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
                self._remember_task_ids(waited.data or {}, canonical)
            records.append(
                {
                    "doc_id": doc.get("id"),
                    "statement_id": doc.get("statement_id") or doc.get("id"),
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
            "id_map_size": len(self._id_map),
        }
        return self.ingest_stats

    def _remember_task_ids(self, payload: dict[str, Any], canonical: str) -> None:
        if not canonical or not isinstance(payload, dict):
            return
        for key in ("chunk_id", "text_chunk_id", "resource_id", "source_chunk_id"):
            self._remember(str(payload.get(key) or ""), canonical, lock=True)
        for key in ("chunk_ids", "text_chunk_ids", "source_chunk_ids", "node_ids"):
            for item in payload.get(key) or []:
                self._remember(str(item), canonical, lock=True)
        result = payload.get("result")
        if isinstance(result, dict):
            self._remember_task_ids(result, canonical)

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
        if isinstance(payload, dict):
            self._retrieve_payloads.append(payload)
        self._index_retrieve_payload(payload)
        passages = self._passages_from_context(payload, question=question)
        pairs: list[tuple[str, str]] = []
        http_calls = 1
        conflict_mode = str(self.capabilities.get("conflict_mode") or "n/a")
        mapping_info: dict[str, Any] = {
            "supported": True,
            "n_raw": 0,
            "n_mapped": 0,
            "n_unmapped": 0,
        }
        conflict_job: dict[str, Any] | None = None
        if question.get("task") == "conflict_audit":
            path = self.settings.conflict_path
            raw_pairs: list[tuple[str, str]] = []
            pair_source = ""
            dedicated_ok = False
            sidecar_ok = "temporal_conflicts" in payload
            if path:
                http_calls += 1
                try:
                    conflict_res = client.request_path(
                        "POST",
                        path,
                        json={
                            "brain_id": self.brain_id,
                            "text": question.get("question"),
                        },
                    )
                except Exception:
                    notes.append("conflict_endpoint_error")
                else:
                    data = conflict_res.data if isinstance(conflict_res.data, dict) else {}
                    if conflict_res.status_code < 400 and _has_pair_collection(data):
                        dedicated_ok = True
                        raw_pairs = self._pairs_from_payload(data)
                        notes.append("conflict_endpoint")
                        if raw_pairs:
                            pair_source = "dedicated_endpoint"
                    elif conflict_res.status_code < 400:
                        notes.append("conflict_endpoint_unrecognized")
                    else:
                        notes.append(f"conflict_endpoint_http_{conflict_res.status_code}")
            if not raw_pairs and sidecar_ok:
                raw_pairs = self._conflict_plugin(payload, question)
                if raw_pairs:
                    notes.append("temporal_conflicts_sidecar")
                    pair_source = "temporal_conflicts_sidecar"
            conflict_job = {
                "raw_pairs": raw_pairs,
                "pair_source": pair_source,
                "producer_ok": dedicated_ok or sidecar_ok,
                "dedicated_ok": dedicated_ok,
                "sidecar_ok": sidecar_ok,
                "notes": notes,
            }
            decided = self._decide_conflict_outcome(conflict_job)
            pairs = decided["pairs"]
            mapping_info = decided["mapping_info"]
            conflict_mode = decided["conflict_mode"]
            self._pending_conflict_jobs.append(conflict_job)
            self._refresh_conflict_capability(conflict_mode)
        self.capabilities["last_as_of_mode"] = as_of_mode
        answer = self._answer(question, passages, payload)
        e2e_ms = (time.perf_counter() - started) * 1000
        result = QueryResult(
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
            raw={
                "context": payload,
                "as_of_mode": as_of_mode,
                "conflict_mode": conflict_mode,
                "conflict_mapping": mapping_info,
                "retrieve_extra_body": extra,
            },
            notes=notes,
        )
        if conflict_job is not None:
            conflict_job["result"] = result
        return result

    def _index_retrieve_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        for key in ("source_passages", "historical_context"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    self._remember_metadata(item)
                    self._remember_text(str(item.get("text") or item.get("body") or item.get("content") or ""))
                else:
                    self._remember_text(str(item))
        self._remember_text(str(payload.get("text_context") or ""))
        for triple in payload.get("triples") or []:
            self._index_triple(triple)
        for key in ("passages", "sources", "documents", "chunks"):
            for row in payload.get(key) or []:
                if isinstance(row, dict):
                    self._remember_metadata(row)

    def _index_triple(self, triple_row: Any) -> None:
        if isinstance(triple_row, dict):
            identified = str(triple_row.get("identified_entity") or "")
            nodes = triple_row.get("triple") or []
            chunk_ids = list(triple_row.get("source_chunk_ids") or [])
        else:
            identified = str(getattr(triple_row, "identified_entity", "") or "")
            nodes = getattr(triple_row, "triple", None) or []
            chunk_ids = list(getattr(triple_row, "source_chunk_ids", None) or [])
        gold = self._canonical(identified)
        parsed = self._remember_text(identified)
        if not gold:
            gold = self._canonical(parsed.get("statement_id")) or self._canonical(parsed.get("doc_id"))
        aliases: list[str] = [str(item) for item in chunk_ids if item]
        for node in nodes:
            parts = _node_parts(node)
            if parts.get("uuid"):
                aliases.append(parts["uuid"])
            name_gold = self._canonical(parts.get("name"))
            if name_gold:
                gold = gold or name_gold
            parsed_name = self._remember_text(parts.get("name") or "")
            if not gold:
                gold = self._canonical(parsed_name.get("statement_id")) or self._canonical(
                    parsed_name.get("doc_id")
                )
            for blob in (parts.get("properties") or {}, parts.get("metadata") or {}):
                if not isinstance(blob, dict):
                    continue
                for key in ("statement_id", "eval_doc_id", "doc_id"):
                    found = self._canonical(str(blob.get(key) or ""))
                    if found:
                        gold = gold or found
        if not gold:
            return
        for alias in aliases:
            self._remember(alias, gold)
        if identified and (identified in self._gold_ids or self._canonical(identified)):
            self._remember(identified, gold)

    def _mapped_conflict_pairs(self, raw_pairs: list[tuple[str, str]]) -> dict[str, Any]:
        return remap_conflict_pairs(raw_pairs, self._id_map)

    def _decide_conflict_outcome(self, job: dict[str, Any]) -> dict[str, Any]:
        raw_pairs = list(job.get("raw_pairs") or [])
        pair_source = str(job.get("pair_source") or "")
        producer_ok = bool(job.get("producer_ok"))
        notes = job.get("notes")
        if isinstance(notes, list):
            notes[:] = [item for item in notes if item not in _REMAP_NOTES]
        mapped = self._mapped_conflict_pairs(raw_pairs)
        mapping_info = {
            "supported": mapped["supported"],
            "n_raw": mapped["n_raw"],
            "n_mapped": mapped["n_mapped"],
            "n_unmapped": mapped["n_unmapped"],
        }
        pairs: list[tuple[str, str]] = []
        if raw_pairs and mapped["n_unmapped"]:
            if isinstance(notes, list):
                notes.append("conflict_mapping_unsupported")
            conflict_mode = "id_mapping_unsupported"
            mapping_info["supported"] = False
            outcome = "unmapped"
        elif mapped["pairs"]:
            pairs = mapped["pairs"]
            conflict_mode = (
                "dedicated_endpoint_mapped"
                if pair_source == "dedicated_endpoint"
                else "temporal_conflicts_mapped"
            )
            if isinstance(notes, list):
                notes.append("conflict_ids_mapped")
            outcome = "mapped"
        elif producer_ok:
            if isinstance(notes, list):
                notes.append("conflict_producer_empty")
            if job.get("dedicated_ok") and pair_source != "temporal_conflicts_sidecar":
                conflict_mode = "dedicated_endpoint"
            else:
                conflict_mode = pair_source or "temporal_conflicts_sidecar"
            mapping_info["supported"] = True
            outcome = "empty"
        else:
            if isinstance(notes, list):
                notes.append("conflict_capability_pending")
            conflict_mode = "pending_product"
            mapping_info["supported"] = False
            outcome = "unavailable"
        job["pairs"] = pairs
        job["mapping_info"] = mapping_info
        job["conflict_mode"] = conflict_mode
        job["outcome"] = outcome
        self._conflict_outcomes.append(outcome)
        result = job.get("result")
        if isinstance(result, QueryResult):
            result.conflict_pairs = pairs
            result.notes = notes if isinstance(notes, list) else result.notes
            raw = dict(result.raw or {})
            raw["conflict_mapping"] = mapping_info
            raw["conflict_mode"] = conflict_mode
            result.raw = raw
        return {
            "pairs": pairs,
            "mapping_info": mapping_info,
            "conflict_mode": conflict_mode,
            "outcome": outcome,
        }

    def finalize_conflict_mappings(self) -> None:
        self._reindex_retrieve_payloads()
        if not self._pending_conflict_jobs:
            return
        self._conflict_outcomes = []
        for job in self._pending_conflict_jobs:
            self._decide_conflict_outcome(job)
        self._refresh_conflict_capability()

    def _reindex_retrieve_payloads(self) -> None:
        previous: tuple[int, int] | None = None
        for _ in range(8):
            snapshot = (len(self._id_map), len(self._ambiguous))
            if snapshot == previous:
                break
            previous = snapshot
            for payload in self._retrieve_payloads:
                self._index_retrieve_payload(payload)

    def _supported_conflict_mode(self) -> str:
        modes: list[str] = []
        for job in self._pending_conflict_jobs:
            if job.get("outcome") not in {"mapped", "empty"}:
                continue
            mode = str(job.get("conflict_mode") or "")
            if mode and mode not in modes:
                modes.append(mode)
        if len(modes) > 1:
            return "mixed"
        if modes:
            return modes[0]
        return "pending_product"

    def _refresh_conflict_capability(self, last_mode: str | None = None) -> None:
        outcomes = set(self._conflict_outcomes)
        supported_mode = self._supported_conflict_mode()
        if "unmapped" in outcomes:
            self.capabilities["conflict_pairs"] = False
            self.capabilities["conflict_mode"] = "id_mapping_unsupported"
            return
        if "mapped" in outcomes or "empty" in outcomes:
            self.capabilities["conflict_pairs"] = True
            self.capabilities["conflict_mode"] = supported_mode
            return
        self.capabilities["conflict_pairs"] = False
        self.capabilities["conflict_mode"] = last_mode or supported_mode or "pending_product"

    def _passages_from_context(
        self,
        payload: dict[str, Any],
        question: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        current = [
            self._passage_record(item, index, prefix="passage")
            for index, item in enumerate(payload.get("source_passages") or [])
        ]
        history = [
            self._passage_record(item, index, prefix="hist")
            for index, item in enumerate(payload.get("historical_context") or [])
        ]
        blob = str(payload.get("text_context") or "").strip()
        if blob and not current and not history:
            current.append(self._passage_record(blob, 0, prefix="text_context"))
        max_n = int(PROMPT_BUDGET["max_passages"])
        if history and _needs_as_of(question or {}):
            hist_budget = max(1, max_n // 2)
            hist_take = min(len(history), hist_budget)
            cur_take = min(len(current), max_n - hist_take)
            leftover = max_n - cur_take - hist_take
            hist_take = min(len(history), hist_take + leftover)
            return current[:cur_take] + history[:hist_take]
        return (current + history)[:max_n]

    def _passage_record(self, raw: Any, index: int, *, prefix: str) -> dict[str, Any]:
        if prefix == "hist":
            default_title = "historical_context"
        elif prefix == "text_context":
            default_title = "text_context"
        elif prefix == "passage":
            default_title = "source_passage"
        else:
            default_title = prefix
        if isinstance(raw, dict):
            text = str(raw.get("text") or raw.get("body") or raw.get("content") or "")
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            title = str(raw.get("title") or metadata.get("title") or default_title)
            raw_id = str(raw.get("id") or f"{prefix}-{index}")
            chunk_id = str(raw.get("chunk_id") or "")
            resolved = self._remember_metadata(raw)
        else:
            text = str(raw)
            metadata = {}
            title = default_title
            raw_id = f"{prefix}-{index}" if prefix != "text_context" else "text_context"
            chunk_id = ""
            resolved = None
        parsed = self._remember_text(text)
        doc_id = resolved
        if isinstance(raw, dict):
            properties = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
            for blob in (raw, metadata, properties):
                if not isinstance(blob, dict):
                    continue
                for key in ("eval_doc_id", "doc_id", "document_id", "statement_id"):
                    doc_id = doc_id or self._canonical(str(blob.get(key) or ""))
            doc_id = doc_id or self._canonical(raw_id) or self._canonical(chunk_id)
        if not doc_id:
            doc_id = self._canonical(parsed.get("statement_id")) or self._canonical(parsed.get("doc_id"))
        if not doc_id:
            doc_id = str(raw_id)
        statement_id = None
        parsed_stmt = parsed.get("statement_id")
        if parsed_stmt and (parsed_stmt in self._gold_ids or self._canonical(parsed_stmt)):
            statement_id = self._canonical(parsed_stmt) or parsed_stmt
        elif doc_id in self._gold_ids:
            statement_id = doc_id
        return {
            "id": str(raw_id),
            "doc_id": str(doc_id),
            "statement_id": statement_id,
            "title": title,
            "text": text,
        }

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
