from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Callable

from mcp.types import CallToolResult
from pydantic import BaseModel

from src.core.intent.brain import BrainResolutionError, resolve_brain
from src.core.intent.format import estimate_tokens
from src.core.intent.models import (
    CheckResult,
    CorrectResult,
    IntentError,
    RecallResult,
    RememberResult,
    ResponseFormat,
)
from src.core.intent.service import check_claim, correct_fact, recall_facts, remember_fact
from src.core.verification.service import FactNotFound
from src.services.mcp.utils import guard_brainpat

RECALL_DESCRIPTION = (
    "Recall what is true now or as of a date. Use for what's true, as of, "
    "what changed, and who said. Returns facts with status. brain_id optional."
)
CHECK_DESCRIPTION = (
    "Check whether a claim is still true. Returns supported, contradicted, "
    "superseded, or unknown with facts and sources. brain_id optional."
)
REMEMBER_DESCRIPTION = (
    "Remember a fact. Writes it unverified. Pass source (document or URL). "
    "source_key rewrites supersede the previous version. Writer comes from the token."
)
CORRECT_DESCRIPTION = (
    "Correct a stored fact without deleting history. Pass fact_id or the fact "
    "text, a reason, and optional correction. Stays on the same entity and predicate."
)

_REGISTERED = False

_READ_ANNOTATIONS = {
    "title": None,
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
_REMEMBER_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
_CORRECT_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _annotations(payload: dict[str, Any]) -> Any:
    try:
        from mcp.types import ToolAnnotations

        return ToolAnnotations(**{k: v for k, v in payload.items() if k != "title" or v is not None})
    except Exception:
        return payload


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if isinstance(model, dict):
        return model
    return {"text": str(model)}


def _result(payload: dict[str, Any]) -> Any:
    text = str(payload.get("text") or payload.get("error") or "")
    try:
        from mcp.types import CallToolResult, TextContent

        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structuredContent=payload,
        )
    except Exception:
        return payload


def _error(model: type[BaseModel], code: str, message: str, hint: str | None = None) -> Any:
    text = message if not hint else f"{message} {hint}"
    fields = set(getattr(model, "model_fields", {}) or {})
    payload: dict[str, Any] = {"text": text, "error": message, "code": code, "hint": hint}
    if "brain_id" in fields:
        payload["brain_id"] = ""
    if "verdict" in fields:
        payload["verdict"] = "unknown"
    if "claim" in fields:
        payload["claim"] = ""
    if "facts" in fields:
        payload["facts"] = []
    if "evidence" in fields:
        payload["evidence"] = []
    if "superseded" in fields:
        payload["superseded"] = []
    try:
        dumped = _dump(model.model_validate(payload))
    except Exception:
        dumped = IntentError(error=message, code=code, hint=hint).as_payload()
        dumped["text"] = text
    return _result(dumped)


def _auth_token() -> str | None:
    from src.services.mcp.main import auth_token_var

    return auth_token_var.get()


def _graph_and_data() -> tuple[Any, Any]:
    from src.core.instances import data_adapter, graph_adapter

    return graph_adapter, data_adapter


def _resolve(brain_id: str | None) -> str:
    return resolve_brain(
        _auth_token(),
        brain_id,
        guard=guard_brainpat,
    )


def _format(value: str | None) -> ResponseFormat:
    text = (value or "concise").strip().lower()
    if text not in {"concise", "detailed"}:
        raise ValueError("response_format must be concise or detailed")
    return text  # type: ignore[return-value]


def _remember_sync(
    fact: str,
    source: str,
    valid_from: str,
    source_key: str,
    brain_id: str,
) -> Any:
    try:
        resolved = _resolve(brain_id)
        graph, data = _graph_and_data()
        result = remember_fact(
            fact=fact,
            source=source,
            brain_id=resolved,
            token=_auth_token(),
            valid_from=valid_from or None,
            source_key=source_key or None,
            graph=graph,
            data=data,
        )
        return _result(_dump(result))
    except BrainResolutionError as exc:
        return _error(RememberResult, exc.code, str(exc), exc.hint)
    except ValueError as exc:
        return _error(RememberResult, "invalid_argument", str(exc))
    except Exception as exc:
        return _error(RememberResult, "remember_failed", f"Could not remember that fact: {exc}")


def _recall_sync(
    question: str,
    as_of: str,
    response_format: str,
    include_rejected: bool,
    limit: int,
    skip: int,
    brain_id: str,
) -> Any:
    try:
        resolved = _resolve(brain_id)
        graph, data = _graph_and_data()
        result = recall_facts(
            question=question,
            brain_id=resolved,
            as_of=as_of or None,
            response_format=_format(response_format),
            include_rejected=include_rejected,
            limit=limit,
            skip=skip,
            graph=graph,
            data=data,
        )
        return _result(_dump(result))
    except BrainResolutionError as exc:
        return _error(RecallResult, exc.code, str(exc), exc.hint)
    except ValueError as exc:
        return _error(RecallResult, "invalid_argument", str(exc))
    except Exception as exc:
        return _error(RecallResult, "recall_failed", f"Could not recall: {exc}")


def _check_sync(claim: str, as_of: str, brain_id: str) -> Any:
    try:
        resolved = _resolve(brain_id)
        graph, data = _graph_and_data()
        result = check_claim(
            claim=claim,
            brain_id=resolved,
            as_of=as_of or None,
            graph=graph,
            data=data,
        )
        return _result(_dump(result))
    except BrainResolutionError as exc:
        return _error(CheckResult, exc.code, str(exc), exc.hint)
    except ValueError as exc:
        return _error(CheckResult, "invalid_argument", str(exc))
    except Exception as exc:
        return _error(CheckResult, "check_failed", f"Could not check that claim: {exc}")


def _correct_sync(
    fact_id: str,
    text: str,
    correction: str,
    reason: str,
    brain_id: str,
) -> Any:
    try:
        resolved = _resolve(brain_id)
        graph, data = _graph_and_data()
        result = correct_fact(
            reason=reason,
            brain_id=resolved,
            token=_auth_token(),
            fact_id=fact_id or None,
            text=text or None,
            correction=correction or None,
            graph=graph,
            data=data,
        )
        return _result(_dump(result))
    except BrainResolutionError as exc:
        return _error(CorrectResult, exc.code, str(exc), exc.hint)
    except FactNotFound:
        return _error(
            CorrectResult,
            "fact_not_found",
            "No matching fact to correct.",
            hint="Pass fact_id from recall, or the exact fact text.",
        )
    except ValueError as exc:
        return _error(CorrectResult, "invalid_argument", str(exc))
    except Exception as exc:
        return _error(CorrectResult, "correct_failed", f"Could not correct that fact: {exc}")


async def brainapi_recall(
    question: str,
    as_of: str = "",
    response_format: str = "concise",
    include_rejected: bool = False,
    limit: int = 8,
    skip: int = 0,
    brain_id: str = "",
) -> Annotated[CallToolResult, RecallResult]:
    """Recall what is true now or as of a date."""
    return await asyncio.to_thread(
        _recall_sync,
        question,
        as_of,
        response_format,
        include_rejected,
        limit,
        skip,
        brain_id,
    )


async def brainapi_check(
    claim: str, as_of: str = "", brain_id: str = ""
) -> Annotated[CallToolResult, CheckResult]:
    """Check whether a claim is still true."""
    return await asyncio.to_thread(_check_sync, claim, as_of, brain_id)


async def brainapi_remember(
    fact: str,
    source: str,
    valid_from: str = "",
    source_key: str = "",
    brain_id: str = "",
) -> Annotated[CallToolResult, RememberResult]:
    """Remember a fact. Lands unverified."""
    return await asyncio.to_thread(
        _remember_sync, fact, source, valid_from, source_key, brain_id
    )


async def brainapi_correct(
    reason: str,
    fact_id: str = "",
    text: str = "",
    correction: str = "",
    brain_id: str = "",
) -> Annotated[CallToolResult, CorrectResult]:
    """Correct a fact without deleting history."""
    return await asyncio.to_thread(
        _correct_sync, fact_id, text, correction, reason, brain_id
    )


def intent_tool_descriptions() -> list[tuple[str, str]]:
    return [
        ("brainapi_recall", RECALL_DESCRIPTION),
        ("brainapi_check", CHECK_DESCRIPTION),
        ("brainapi_remember", REMEMBER_DESCRIPTION),
        ("brainapi_correct", CORRECT_DESCRIPTION),
    ]


def intent_tool_token_count() -> int:
    blob = "\n".join(f"{name}: {description}" for name, description in intent_tool_descriptions())
    blob += "\n" + json.dumps(
        {
            "brainapi_recall": ["question", "as_of", "response_format", "include_rejected", "limit", "skip", "brain_id"],
            "brainapi_check": ["claim", "as_of", "brain_id"],
            "brainapi_remember": ["fact", "source", "valid_from", "source_key", "brain_id"],
            "brainapi_correct": ["reason", "fact_id", "text", "correction", "brain_id"],
        }
    )
    return estimate_tokens(blob)


def register_intent_tools(mcp_server: Any, register: Callable | None = None) -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    if register is not None:
        def binder(**kwargs: Any):
            def deco(fn: Callable) -> Callable:
                register(fn, **kwargs)
                return fn

            return deco
    else:
        binder = getattr(mcp_server, "tool", None)
    if binder is None:
        return

    def _bind(fn, *, name: str, description: str, annotations: dict[str, Any]) -> None:
        binder(
            name=name,
            description=description,
            annotations=_annotations(annotations),
            structured_output=True,
        )(fn)

    _bind(
        brainapi_recall,
        name="brainapi_recall",
        description=RECALL_DESCRIPTION,
        annotations=_READ_ANNOTATIONS,
    )
    _bind(
        brainapi_check,
        name="brainapi_check",
        description=CHECK_DESCRIPTION,
        annotations=_READ_ANNOTATIONS,
    )
    _bind(
        brainapi_remember,
        name="brainapi_remember",
        description=REMEMBER_DESCRIPTION,
        annotations=_REMEMBER_ANNOTATIONS,
    )
    _bind(
        brainapi_correct,
        name="brainapi_correct",
        description=CORRECT_DESCRIPTION,
        annotations=_CORRECT_ANNOTATIONS,
    )
    _REGISTERED = True
