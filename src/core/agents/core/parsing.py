import json
import re
from types import SimpleNamespace
from typing import Any, Optional, Type

from pydantic import BaseModel

from src.utils.cleanup import _last_json_object, _repair_trailing_commas, strip_json


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        part = content.get("text") or content.get("content")
        if part is not None:
            if isinstance(part, str):
                return part
            if isinstance(part, list):
                return normalize_message_content(part)
            if isinstance(part, dict):
                return normalize_message_content(part)
        if "parts" in content:
            return normalize_message_content(content["parts"])
        return json.dumps(content) if content else ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("content")
                if part is None and "json" in block:
                    part = json.dumps(block["json"])
                if part is not None:
                    if isinstance(part, str):
                        parts.append(part)
                    else:
                        parts.append(normalize_message_content(part))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


def normalize_tool_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().rstrip(":")
    return s if s else None


def parse_structured_from_messages(
    messages: list, output_schema: Type[BaseModel]
) -> Optional[BaseModel]:
    if not messages or output_schema is None:
        return None
    last = messages[-1]
    content = (
        last.get("content")
        if isinstance(last, dict)
        else getattr(last, "content", None)
    )
    raw = normalize_message_content(content)
    if not raw or not raw.strip():
        return None
    parsed = strip_json(raw)
    if not parsed:
        return None
    try:
        return output_schema.model_validate(parsed)
    except Exception:
        return None


def get_tool_call_from_response(
    response: Any,
) -> tuple[Optional[str], Optional[Any]]:
    if response is None:
        return None, None
    content = normalize_message_content(
        response.get("content")
        if isinstance(response, dict)
        else getattr(response, "content", None)
    )
    parsed = strip_json(content) if content else {}
    if parsed.get("tool_name") is not None:
        return normalize_tool_name(parsed.get("tool_name")), parsed.get("tool_input")
    tool_calls = (
        response.get("tool_calls")
        if isinstance(response, dict)
        else getattr(response, "tool_calls", None)
    ) or []
    if tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            name = first.get("name") or (first.get("function") or {}).get("name")
            args = first.get("args")
            if args is None and first.get("function"):
                raw = (first["function"] or {}).get("arguments")
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw or {}
            args = args or {}
            if args.get("tool_name") is not None:
                return normalize_tool_name(args.get("tool_name")), args.get(
                    "tool_input"
                )
            return normalize_tool_name(name), args if args else None
        name = getattr(first, "name", None) or (
            getattr(getattr(first, "function", None), "name", None)
        )
        args = getattr(first, "args", None)
        if args is None and hasattr(first, "function"):
            func = getattr(first, "function", None)
            if func is not None:
                raw = getattr(func, "arguments", None)
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw or {}
        args = args if isinstance(args, dict) else {}
        if args and args.get("tool_name") is not None:
            return normalize_tool_name(args.get("tool_name")), args.get("tool_input")
        return normalize_tool_name(name), args if args else None
    return None, None


def get_tool_call_from_malformed_response(
    response: Any,
) -> tuple[Optional[str], Optional[Any]]:
    if response is None:
        return None, None
    metadata = (
        response.get("response_metadata")
        if isinstance(response, dict)
        else getattr(response, "response_metadata", None)
    ) or {}
    if metadata.get("finish_reason") != "MALFORMED_FUNCTION_CALL":
        return None, None
    msg = metadata.get("finish_message") or ""
    if "call:default_api:" not in msg or "{" not in msg:
        return None, None
    try:
        rest = msg.split("call:default_api:", 1)[1].strip()
        tool_name_part, _, args_part = rest.partition("{")
        tool_name = tool_name_part.strip().rstrip(":")
        if not tool_name:
            return None, None
        args_str = "{" + args_part
        quoted = re.sub(
            r"([{\[,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
            r'\1"\2":',
            args_str,
        )
        quoted = quoted.replace(r"\'", "'")
        parsed = strip_json(quoted) or _last_json_object(quoted)
        if not parsed and args_str.strip() != "{":
            parsed = strip_json(args_str) or _last_json_object(args_str)
        return tool_name, parsed if parsed else {}
    except Exception:
        return None, None


def get_first_tool_call_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    tool_calls = (
        response.get("tool_calls")
        if isinstance(response, dict)
        else getattr(response, "tool_calls", None)
    ) or []
    if not tool_calls:
        return None
    first = tool_calls[0]
    if isinstance(first, dict):
        return first.get("id") or (first.get("function") or {}).get("id")
    return getattr(first, "id", None)


def content_breaks_ollama_tool_parse(content: Any) -> bool:
    if not content or not isinstance(content, str):
        return False
    return "<|" in content


def get_thought_signature_from_response(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        ak = response.get("additional_kwargs") or {}
        sig_map = ak.get("__gemini_function_call_thought_signatures__")
        if isinstance(sig_map, dict) and sig_map:
            first = next(iter(sig_map.values()), None)
            if first is not None:
                return (
                    str(first)
                    if not isinstance(first, bytes)
                    else first.decode("utf-8")
                )
        out = ak.get("thought_signature") or ak.get("thoughtSignature")
        if out is not None:
            return str(out) if not isinstance(out, bytes) else out.decode("utf-8")
        content = response.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ts = part.get("thoughtSignature") or part.get("thought_signature")
                    if ts is not None:
                        return (
                            str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
                        )
        tool_calls = response.get("tool_calls") or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                ec = (tc.get("extra_content") or {}).get("google") or {}
                ts = ec.get("thought_signature")
                if ts is not None:
                    return str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
        return None
    ak = getattr(response, "additional_kwargs", None) or {}
    sig_map = ak.get("__gemini_function_call_thought_signatures__")
    if isinstance(sig_map, dict) and sig_map:
        first = next(iter(sig_map.values()), None)
        if first is not None:
            return str(first) if not isinstance(first, bytes) else first.decode("utf-8")
    out = ak.get("thought_signature") or ak.get("thoughtSignature")
    if out is not None:
        return str(out) if not isinstance(out, bytes) else out.decode("utf-8")
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ts = part.get("thoughtSignature") or part.get("thought_signature")
                if ts is not None:
                    return str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
    tool_calls = getattr(response, "tool_calls", None) or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            ec = (tc.get("extra_content") or {}).get("google") or {}
            ts = ec.get("thought_signature")
            if ts is not None:
                return str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
        else:
            ec = (getattr(tc, "extra_content", None) or {}).get("google") or {}
            ts = ec.get("thought_signature")
            if ts is not None:
                return str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
    return None


def get_thought_signatures_by_tool_call(response: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if response is None:
        return result
    ak = (
        response.get("additional_kwargs")
        if isinstance(response, dict)
        else getattr(response, "additional_kwargs", None)
    ) or {}
    sig_map = ak.get("__gemini_function_call_thought_signatures__")
    if isinstance(sig_map, dict) and sig_map:
        for tid, sig in sig_map.items():
            if sig is not None:
                result[str(tid)] = sig if isinstance(sig, str) else sig.decode("utf-8")
        if result:
            return result
    tool_calls = (
        response.get("tool_calls")
        if isinstance(response, dict)
        else getattr(response, "tool_calls", None)
    ) or []
    for i, tc in enumerate(tool_calls):
        tid = None
        sig = None
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            tid = tc.get("id") or (fn.get("id") if isinstance(fn, dict) else None)
            ec = (tc.get("extra_content") or {}).get("google") or {}
            sig = ec.get("thought_signature")
        else:
            tid = getattr(tc, "id", None)
            ec = (getattr(tc, "extra_content", None) or {}).get("google") or {}
            sig = ec.get("thought_signature")
        if tid and sig is not None:
            result[tid] = sig if isinstance(sig, str) else sig.decode("utf-8")
    if result:
        return result
    content = (
        response.get("content")
        if isinstance(response, dict)
        else getattr(response, "content", None)
    )
    if isinstance(content, list) and tool_calls:
        tc_index = 0
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("functionCall")
                and tc_index < len(tool_calls)
            ):
                ts = part.get("thoughtSignature") or part.get("thought_signature")
                if ts is not None:
                    sig_str = ts if isinstance(ts, str) else ts.decode("utf-8")
                    tc = tool_calls[tc_index]
                    tid = None
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        tid = tc.get("id") or (
                            fn.get("id") if isinstance(fn, dict) else None
                        )
                    else:
                        tid = getattr(tc, "id", None)
                    if tid:
                        result[tid] = sig_str
                    tc_index += 1
        if result:
            return result
    single = get_thought_signature_from_response(response)
    if single is not None:
        for tc in tool_calls:
            tid = None
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                tid = tc.get("id") or (fn.get("id") if isinstance(fn, dict) else None)
            else:
                tid = getattr(tc, "id", None)
            if tid:
                result[tid] = single
    return result


def serialize_tool_calls(raw_tc: list, counter: int) -> list[dict]:
    def _one(tc: Any, i: int) -> dict:
        base = {}
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args = tc.get("args")
            if args is None and isinstance(fn, dict) and fn.get("arguments"):
                try:
                    args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    args = {}
            base = {
                "id": tc.get("id")
                or (fn.get("id") if isinstance(fn, dict) else None)
                or f"call_{counter + i}",
                "name": tc.get("name")
                or (fn.get("name") if isinstance(fn, dict) else None),
                "args": args or {},
            }
        else:
            base = {
                "id": getattr(tc, "id", None) or f"call_{counter + i}",
                "name": getattr(tc, "name", None)
                or getattr(getattr(tc, "function", None), "name", None),
                "args": getattr(tc, "args", None) or {},
            }
        args = base.get("args") or {}
        if isinstance(args, dict) and (
            not base.get("name") or not str(base.get("name", "")).strip()
        ):
            base["name"] = args.get("tool_name") or base.get("name")
        if not base.get("name") or not str(base.get("name", "")).strip():
            base["name"] = "unknown_tool"
        base["name"] = normalize_tool_name(base["name"]) or base["name"]
        return base

    return [_one(tc, i) for i, tc in enumerate(raw_tc)]


def get_reasoning_from_response(msg: Any) -> Optional[str]:
    if msg is None:
        return None
    if isinstance(msg, dict):
        reasoning = msg.get("thinking") or (
            (msg.get("additional_kwargs") or {}).get("thinking")
        )
        return str(reasoning).strip() if reasoning else None
    reasoning = getattr(msg, "thinking", None) or (
        (getattr(msg, "additional_kwargs", None) or {}).get("thinking")
    )
    if reasoning:
        return str(reasoning).strip()
    metadata = getattr(msg, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        reasoning = metadata.get("thinking") or metadata.get("reasoning_content")
        return str(reasoning).strip() if reasoning else None
    return None


def get_finish_reason_from_response(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        metadata = response.get("response_metadata") or {}
        if isinstance(metadata, dict) and metadata.get("finish_reason"):
            return metadata.get("finish_reason")
        return (response.get("additional_kwargs") or {}).get("finish_reason")
    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("finish_reason"):
        return metadata.get("finish_reason")
    return (getattr(response, "additional_kwargs", None) or {}).get("finish_reason")


def normalize_invoke_response(response: Any) -> Any:
    if response is None:
        return None
    generations = (
        getattr(response, "generations", None)
        if not isinstance(response, dict)
        else response.get("generations")
    )
    if not generations or not isinstance(generations, list) or not generations:
        return response
    first_chain = generations[0]
    if not first_chain or not isinstance(first_chain, list):
        return response
    gen = first_chain[0]
    if gen is None:
        return response
    message = (
        getattr(gen, "message", None)
        if not isinstance(gen, dict)
        else gen.get("message")
    )
    if isinstance(message, dict) and "kwargs" in message:
        message = message["kwargs"]
    usage = None
    if isinstance(gen, dict):
        usage = gen.get("usage_metadata")
        if usage is None:
            gi = gen.get("generation_info") or {}
            usage = gi.get("usage_metadata")
            if (
                isinstance(usage, dict)
                and "input_tokens" not in usage
                and "prompt_token_count" in usage
            ):
                usage = {
                    "input_tokens": usage.get("prompt_token_count", 0),
                    "output_tokens": usage.get("candidates_token_count", 0),
                    "total_tokens": usage.get("total_token_count", 0),
                }
    else:
        usage = getattr(gen, "usage_metadata", None)
        if usage is None and hasattr(gen, "generation_info"):
            gi = getattr(gen, "generation_info", None) or {}
            usage = (
                gi.get("usage_metadata")
                if isinstance(gi, dict)
                else getattr(gi, "usage_metadata", None)
            )
            if (
                isinstance(usage, dict)
                and "input_tokens" not in usage
                and "prompt_token_count" in usage
            ):
                usage = {
                    "input_tokens": usage.get("prompt_token_count", 0),
                    "output_tokens": usage.get("candidates_token_count", 0),
                    "total_tokens": usage.get("total_token_count", 0),
                }
    if message is None and isinstance(gen, dict):
        content = gen.get("text") or gen.get("content") or ""
        message = {
            "content": content,
            "tool_calls": gen.get("tool_calls") or [],
            "response_metadata": gen.get("response_metadata") or {},
            "additional_kwargs": gen.get("additional_kwargs") or {},
        }
    if message is None:
        return response
    if isinstance(message, dict):
        msg_usage = message.get("usage_metadata")
        if msg_usage is None and "response_metadata" in message:
            rm = message.get("response_metadata") or {}
            msg_usage = rm.get("usage_metadata") if isinstance(rm, dict) else None
        if usage and not msg_usage:
            message = dict(message)
            message["usage_metadata"] = usage
        elif (
            msg_usage
            and isinstance(msg_usage, dict)
            and "input_tokens" not in msg_usage
            and "prompt_token_count" in msg_usage
        ):
            message = dict(message)
            message["usage_metadata"] = {
                "input_tokens": msg_usage.get("prompt_token_count", 0),
                "output_tokens": msg_usage.get("candidates_token_count", 0),
                "total_tokens": msg_usage.get("total_token_count", 0),
            }
    else:
        if usage and not getattr(message, "usage_metadata", None):
            try:
                setattr(message, "usage_metadata", usage)
            except Exception:
                pass
    if isinstance(message, dict):
        return SimpleNamespace(**message)
    return message
