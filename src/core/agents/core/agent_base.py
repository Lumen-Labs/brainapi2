"""
File: /agent_base.py
Project: core
Created Date: Wednesday February 25th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday February 25th 2026 8:26:28 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
import re
import warnings
from typing import Any, Dict, Literal, Optional, Type, TypedDict, get_args, get_origin

import langsmith
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, TypeAdapter, ValidationError

from src.utils.cleanup import strip_json, _last_json_object, _repair_trailing_commas

warnings.filterwarnings(
    "ignore",
    message=r"Unrecognized FinishReason enum value",
    category=UserWarning,
)


def _normalize_message_content(content: Any) -> str:
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
                return _normalize_message_content(part)
            if isinstance(part, dict):
                return _normalize_message_content(part)
        if "parts" in content:
            return _normalize_message_content(content["parts"])
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
                        parts.append(_normalize_message_content(part))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


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
    raw = _normalize_message_content(content)
    if not raw or not raw.strip():
        return None
    parsed = strip_json(raw)
    if not parsed:
        return None
    try:
        return output_schema.model_validate(parsed)
    except Exception:
        return None


def _get_tool_call_from_response(response: Any) -> tuple[Optional[str], Optional[Any]]:
    if response is None:
        return None, None
    content = _normalize_message_content(
        response.get("content")
        if isinstance(response, dict)
        else getattr(response, "content", None)
    )
    parsed = strip_json(content) if content else {}
    if parsed.get("tool_name") is not None:
        return parsed.get("tool_name"), parsed.get("tool_input")
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
                return args.get("tool_name"), args.get("tool_input")
            return name, args if args else None
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
            return args.get("tool_name"), args.get("tool_input")
        return name, args if args else None
    return None, None


def _get_tool_call_from_malformed_response(
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


def _get_first_tool_call_id(response: Any) -> Optional[str]:
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


def _content_breaks_ollama_tool_parse(content: Any) -> bool:
    if not content or not isinstance(content, str):
        return False
    return "<|" in content


def _get_thought_signature_from_response(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        ak = response.get("additional_kwargs") or {}
        sig_map = ak.get("__gemini_function_call_thought_signatures__")
        if isinstance(sig_map, dict) and sig_map:
            first = next(iter(sig_map.values()), None)
            if first is not None:
                return str(first) if not isinstance(first, bytes) else first.decode("utf-8")
        out = ak.get("thought_signature") or ak.get("thoughtSignature")
        if out is not None:
            return str(out) if not isinstance(out, bytes) else out.decode("utf-8")
        content = response.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ts = part.get("thoughtSignature") or part.get("thought_signature")
                    if ts is not None:
                        return str(ts) if not isinstance(ts, bytes) else ts.decode("utf-8")
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


def _get_thought_signatures_by_tool_call(response: Any) -> dict[str, str]:
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
            if isinstance(part, dict) and part.get("functionCall") and tc_index < len(tool_calls):
                ts = part.get("thoughtSignature") or part.get("thought_signature")
                if ts is not None:
                    sig_str = ts if isinstance(ts, str) else ts.decode("utf-8")
                    tc = tool_calls[tc_index]
                    tid = None
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        tid = tc.get("id") or (fn.get("id") if isinstance(fn, dict) else None)
                    else:
                        tid = getattr(tc, "id", None)
                    if tid:
                        result[tid] = sig_str
                    tc_index += 1
        if result:
            return result
    single = _get_thought_signature_from_response(response)
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


def _serialize_tool_calls(raw_tc: list, counter: int) -> list[dict]:
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
        if isinstance(args, dict) and (not base.get("name") or not str(base.get("name", "")).strip()):
            base["name"] = args.get("tool_name") or base.get("name")
        if not base.get("name") or not str(base.get("name", "")).strip():
            base["name"] = "unknown_tool"
        return base

    return [_one(tc, i) for i, tc in enumerate(raw_tc)]


def _get_reasoning_from_response(msg: Any) -> Optional[str]:
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


def _get_finish_reason_from_response(response: Any) -> Optional[str]:
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


def _flatten_json_schema_for_llm(schema: dict) -> dict:
    def resolve_refs(obj: dict, defs: dict) -> dict:
        out = {}
        for k, v in obj.items():
            if k == "description":
                continue
            if k == "$ref":
                ref = v.split("/")[-1]
                return resolve_refs(defs.get(ref, {}).copy(), defs)
            if isinstance(v, dict):
                out[k] = resolve_refs(v, defs)
            elif isinstance(v, list):
                out[k] = [
                    resolve_refs(x, defs) if isinstance(x, dict) else x for x in v
                ]
            else:
                out[k] = v
        return out

    def drop_internal_keys(obj: dict) -> dict:
        out = {}
        for k, v in obj.items():
            if k in ("$defs", "$ref", "strict"):
                continue
            if k == "title" and isinstance(v, str) and v.startswith("_"):
                continue
            if isinstance(v, dict):
                out[k] = drop_internal_keys(v)
            elif isinstance(v, list):
                out[k] = [
                    drop_internal_keys(x) if isinstance(x, dict) else x for x in v
                ]
            else:
                out[k] = v
        return out

    copy = dict(schema)
    defs = copy.pop("$defs", {})
    resolved = resolve_refs(copy, defs)
    return drop_internal_keys(resolved)


def _get_single_list_field_name(schema: Any) -> Optional[str]:
    if schema is None:
        return None
    if hasattr(schema, "model_fields"):
        for name, field in schema.model_fields.items():
            ann = getattr(field, "annotation", None)
            if get_origin(ann) is list:
                return name
    try:
        if hasattr(schema, "model_json_schema"):
            js = schema.model_json_schema()
        else:
            js = TypeAdapter(schema).json_schema()
    except (Exception, NameError):
        return None
    defs = js.get("$defs") or {}
    props = js.get("properties") or {}
    if len(props) != 1:
        return None
    for name, prop in props.items():
        if isinstance(prop, dict) and prop.get("type") == "array":
            return name
        if isinstance(prop, dict) and "$ref" in prop:
            ref = prop["$ref"].split("/")[-1]
            ref_schema = defs.get(ref, {})
            if isinstance(ref_schema, dict) and ref_schema.get("type") == "array":
                return name
    return None


class AgentMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class AgentOutput(TypedDict):
    messages: list
    structured_response: Optional[BaseModel]


class MessagesDict(TypedDict):
    messages: list[AgentMessage]


class AgentBase:
    model: BaseChatModel
    system_prompt: str
    tools: list[BaseTool]
    output_schema: Optional[BaseModel]
    debug: bool

    thinking: bool

    messages: list[AgentMessage]

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        tools: list[BaseTool],
        output_schema: Optional[BaseModel] = None,
        debug: bool = False,
        thinking: bool = False,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.output_schema = output_schema
        self.messages = []
        self.debug = debug
        self.thinking = thinking

        if self.thinking:
            self.model.extra_body = {
                **self.model.extra_body,
                "think": (
                    ("high" if self.thinking else "low")
                    if "gpt" in self.model and "oss" in self.model
                    else (True if self.thinking else False)
                ),
            }

    def _model_requires_thought_signatures(self) -> bool:
        mod = getattr(type(self.model), "__module__", "") or ""
        name = getattr(type(self.model), "__name__", "") or ""
        return "vertex" in mod.lower() or "Vertex" in name

    def _get_effective_output_schema(self):
        if self.output_schema is None:
            return None
        effective = getattr(self.output_schema, "schema", self.output_schema)
        if callable(effective):
            effective = self.output_schema
        return effective

    def _get_output_schema_json_schema(self) -> Optional[dict]:
        effective = self._get_effective_output_schema()
        if effective is None:
            return None
        try:
            if hasattr(effective, "model_json_schema"):
                return effective.model_json_schema()
            return TypeAdapter(effective).json_schema()
        except (Exception, NameError):
            return None  # "low", "medium", "high" for gpt-oss

    def _validate_list_response_fallback(
        self, effective: Type[BaseModel], list_field: str, items: list
    ) -> Optional[BaseModel]:
        if not getattr(effective, "model_fields", None):
            return None
        field_info = effective.model_fields.get(list_field)
        if field_info is None:
            return None
        ann = getattr(field_info, "annotation", None)
        if get_origin(ann) is not list:
            return None
        args = get_args(ann)
        if not args:
            return None
        item_type = args[0]
        if not isinstance(item_type, type):
            return None
        validated = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                if hasattr(item_type, "model_validate"):
                    validated.append(item_type.model_validate(item))
                else:
                    validated.append(TypeAdapter(item_type).validate_python(item))
            except Exception:
                try:
                    allowed = set(getattr(item_type, "model_fields", {}).keys())
                    filtered = {k: v for k, v in item.items() if k in allowed}
                    if allowed and filtered:
                        if hasattr(item_type, "model_validate"):
                            validated.append(item_type.model_validate(filtered))
                        else:
                            validated.append(TypeAdapter(item_type).validate_python(filtered))
                except Exception:
                    pass
        if not validated:
            return None
        try:
            return effective.model_validate({list_field: validated})
        except Exception:
            try:
                return effective.model_construct(**{list_field: validated})
            except Exception:
                return None

    def _normalize_tool_input(self, tool: BaseTool, raw_input: Any) -> str | dict:
        if isinstance(raw_input, (dict, str)):
            return raw_input
        if isinstance(raw_input, list):
            schema = getattr(tool, "args_schema", None)
            if isinstance(schema, dict):
                required = schema.get("required") or []
                properties = schema.get("properties") or {}
                for key in required:
                    if (
                        isinstance(properties.get(key), dict)
                        and properties[key].get("type") == "array"
                    ):
                        return {key: raw_input}
            return {"input": raw_input}
        return {"input": raw_input}

    def _call_tool(self, tool: BaseTool, input: Any):
        normalized = self._normalize_tool_input(tool, input)
        if self.debug:
            print(
                f"[DEBUG (agent_base)]: calling tool {tool.name} with input: {normalized}"
            )
        return tool.run(normalized)

    def _format_tool_description(self, tool: BaseTool):
        args = getattr(tool, "args_schema", None)
        if hasattr(args, "model_json_schema"):
            input_str = json.dumps(args.model_json_schema(), indent=2)
        elif isinstance(args, dict):
            input_str = json.dumps(args, indent=2)
        else:
            input_str = str(args) if args is not None else "N/A"
        return f"""
        {tool.name}: {tool.description}
        Input schema: {input_str}
        """

    def _system_internal_prompt(self):
        tools_block = ""
        if self.tools:
            tools_list = "\n".join(self._format_tool_description(t) for t in self.tools)
            tools_block = f"""
        You are able to use the following tools:
        {tools_list}

        The input to those tools if required must be ONLY the input specified in the tool's info.

        To call a tool you can use your tool calling caapability or output a JSON object with the following structure:
        ```json
        {{
            "tool_name": "tool_name",
            "tool_input": "tool_input"
        }}
        ```
        {"You MUST NOT use your internal function calling capability, if you want to use a tool just output the json above and nothing else, the system will recall you back to continue the workflow with the output of the tool." if 'gpt' in self.model and 'oss' in self.model else ''}
        
        Proceed ONLY one tool at a time, when requesting a tool you must not use more than one tool in the same output, after requesting a tool
        you will be recalled back by the system to continue the workflow.
        """
        output_schema_block = ""
        if self.output_schema:
            schema_dict = self._get_output_schema_json_schema()
            if schema_dict is not None:
                output_schema_block = f"""
        Your output must be a JSON object ONLY with the following JSON schema structure:
        {json.dumps(_flatten_json_schema_for_llm(schema_dict), indent=2)}
        You must strictly follow the schema since it's a JSON Schema Specification, don't add any additional fields or properties.
        Don't output the same fields of the JSON schema as it is, it's just the schema instructions, you must output the actual data.
        Remember that the fields that you are interested in are inside the 'properties' key of the JSON schema.
        You must return the JSON ONLY, no additional text or comments or explanation.
        """
        return f"""
        You are a helpful agent. You must follow the instructions given to you by the user strictly.
        {tools_block}
        {output_schema_block}
        {"/no_think" if not self.thinking else ""}
        """

    def _parse_tool_for_llm_input(self, tool: BaseTool):
        return f"""
        {tool.name}: {tool.description}
        Input: {tool.args_schema}
        """

    def invoke(self, messages: MessagesDict, config: Optional[dict] = None):

        config = config or {}
        project_name = os.getenv("LANGSMITH_PROJECT", "brainapi")
        tags = list(config.get("tags", [])) + ["agent_base"]
        metadata = dict(config.get("metadata", {}))

        with langsmith.tracing_context(
            project_name=project_name,
            enabled=True,
            tags=tags,
            metadata=metadata,
        ):
            return self._invoke_impl(messages.get("messages"), config)

    def _invoke_impl(self, messages: list[AgentMessage], config: dict):

        model_responses: list = []
        self.messages = []
        self.messages.append(
            {
                "role": "system",
                "content": self._system_internal_prompt() + "\n" + self.system_prompt,
            }
        )
        self.messages.extend(messages)
        tool_call_id_counter = 0

        if self.debug:
            tags = (config or {}).get("tags", [])
            print(
                "[DEBUG (agent_base)]: Invoking agent with messages: ",
                self.messages,
                "tags:",
                tags,
            )

        n_message = None
        structured_response = None
        while True:
            _did_retry_recovered_tool_call = False
            while n_message is None:
                _did_retry_unknown_finish = False
                _invoke_attempts = 3
                for _invoke_attempt in range(_invoke_attempts):
                    _n_message = self.model.invoke(self.messages, config)
                    model_responses.append(_n_message)
                raw_content = _normalize_message_content(
                    _n_message.get("content")
                    if isinstance(_n_message, dict)
                    else getattr(_n_message, "content", None)
                )
                if self.debug:
                    print("[DEBUG (agent_base)]: Raw content: ", raw_content)
                    reasoning = _get_reasoning_from_response(_n_message)
                    if reasoning:
                        print("[DEBUG (agent_base)]: Reasoning: ", reasoning)
                n_message = raw_content
                tool_call_id = (
                    _get_first_tool_call_id(_n_message)
                    or f"call_{tool_call_id_counter}"
                )
                assistant_msg: dict = {"role": "assistant", "content": n_message}
                raw_tc = (
                    getattr(_n_message, "tool_calls", None)
                    or (isinstance(_n_message, dict) and _n_message.get("tool_calls"))
                    or []
                )
                if raw_tc:
                    msg_thought_sig = _get_thought_signature_from_response(_n_message)
                    assistant_msg["tool_calls"] = _serialize_tool_calls(
                        raw_tc,
                        tool_call_id_counter,
                    )
                    assistant_msg["content"] = ""
                    sig_by_id = _get_thought_signatures_by_tool_call(_n_message)
                    if not sig_by_id and msg_thought_sig is not None:
                        for tc in assistant_msg["tool_calls"]:
                            tid = tc.get("id")
                            if tid:
                                sig_by_id[tid] = msg_thought_sig
                    if sig_by_id:
                        sig_map = assistant_msg.setdefault("additional_kwargs", {})
                        existing = sig_map.get(
                            "__gemini_function_call_thought_signatures__", {}
                        )
                        if not isinstance(existing, dict):
                            existing = {}
                        for tc in assistant_msg["tool_calls"]:
                            tid = tc.get("id")
                            if tid:
                                existing[tid] = sig_by_id.get(tid) or msg_thought_sig
                        sig_map["__gemini_function_call_thought_signatures__"] = existing
                        break
                    if _invoke_attempt >= _invoke_attempts - 1:
                        raise ValueError(
                            "Model returned tool calls but no thought signatures; "
                            "Vertex AI requires thought signatures for function calls. "
                            "Retries exhausted."
                        )
                    continue
                break
            if raw_tc:
                pass
            elif _content_breaks_ollama_tool_parse(n_message):
                assistant_msg["content"] = ""
            _recovered_name, _recovered_input = _get_tool_call_from_response(
                _n_message
            )
            if _recovered_name is None and n_message:
                _parsed = strip_json(n_message)
                _recovered_name = _parsed.get("tool_name")
                _recovered_input = _parsed.get("tool_input")
            if _recovered_name is None:
                _recovered_name, _recovered_input = (
                    _get_tool_call_from_malformed_response(_n_message)
                )
            if not raw_tc and _recovered_name is not None:
                if not _did_retry_recovered_tool_call:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Please use your native tool calling capability to call the tool.",
                        }
                    )
                    _did_retry_recovered_tool_call = True
                    n_message = None
                    continue
                tool = next(
                    (t for t in self.tools if t.name == _recovered_name),
                    None,
                )
                if tool is not None:
                    n_message = self._call_tool(tool, _recovered_input)
                    if self._model_requires_thought_signatures():
                        self.messages.append(
                            {
                                "role": "user",
                                "content": f"You called the tool `{_recovered_name}`. The tool returned:\n{n_message}\n\nContinue with the next step.",
                            }
                        )
                    else:
                        recovered_id = f"call_{tool_call_id_counter}"
                        recovered_tc = {
                            "id": recovered_id,
                            "name": _recovered_name,
                            "args": (
                                _recovered_input
                                if isinstance(_recovered_input, dict)
                                else (
                                    {"input": _recovered_input}
                                    if _recovered_input is not None
                                    else {}
                                )
                            ),
                            "extra_content": {"google": {"thought_signature": ""}},
                        }
                        recovered_assistant_msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [recovered_tc],
                        }
                        recovered_assistant_msg.setdefault("additional_kwargs", {})[
                            "__gemini_function_call_thought_signatures__"
                        ] = {recovered_id: ""}
                        self.messages.append(recovered_assistant_msg)
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": n_message,
                                "tool_call_id": recovered_id,
                            }
                        )
                    tool_call_id_counter += 1
                    n_message = None
                    continue
            self.messages.append(assistant_msg)
            tool_name, tool_input = _recovered_name, _recovered_input
            if tool_name is None:
                tool_name, tool_input = _get_tool_call_from_response(_n_message)
            if tool_name is None and n_message:
                parsed = strip_json(n_message)
                tool_name = parsed.get("tool_name")
                tool_input = parsed.get("tool_input")
            if tool_name is None:
                tool_name, tool_input = _get_tool_call_from_malformed_response(
                    _n_message
                )
            if tool_name is not None:
                tool = next(
                    (t for t in self.tools if t.name == tool_name),
                    None,
                )
                if tool is not None:
                    n_message = self._call_tool(tool, tool_input)
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": n_message,
                            "tool_call_id": tool_call_id,
                        }
                    )
                    tool_call_id_counter += 1
                    _did_retry_recovered_next_tool_call = False
                    while True:
                        _next_attempts = 3
                        for _next_attempt in range(_next_attempts):
                            next_response = self.model.invoke(self.messages, config)
                            if self.debug:
                                print(
                                    "[DEBUG (agent_base)]: Next response: ",
                                    next_response,
                                )
                                reasoning = _get_reasoning_from_response(next_response)
                                if reasoning:
                                    print(
                                        "[DEBUG (agent_base)]: Reasoning: ",
                                        reasoning,
                                    )
                            model_responses.append(next_response)
                            next_content = _normalize_message_content(
                                next_response.get("content")
                                if isinstance(next_response, dict)
                                else getattr(next_response, "content", None)
                            )
                            next_tool_call_id = (
                                _get_first_tool_call_id(next_response)
                                or f"call_{tool_call_id_counter}"
                            )
                            next_assistant_msg = {
                                "role": "assistant",
                                "content": next_content,
                            }
                            next_raw_tc = (
                                getattr(next_response, "tool_calls", None)
                                or (
                                    isinstance(next_response, dict)
                                    and next_response.get("tool_calls")
                                )
                                or []
                            )
                            if next_raw_tc:
                                next_msg_thought_sig = (
                                    _get_thought_signature_from_response(next_response)
                                )
                                next_assistant_msg["tool_calls"] = (
                                    _serialize_tool_calls(
                                        next_raw_tc,
                                        tool_call_id_counter,
                                    )
                                )
                                next_assistant_msg["content"] = ""
                                next_sig_by_id = _get_thought_signatures_by_tool_call(
                                    next_response
                                )
                                if not next_sig_by_id and next_msg_thought_sig is not None:
                                    for tc in next_assistant_msg["tool_calls"]:
                                        tid = tc.get("id")
                                        if tid:
                                            next_sig_by_id[tid] = next_msg_thought_sig
                                if next_msg_thought_sig is not None:
                                    sig_map = next_assistant_msg.setdefault(
                                        "additional_kwargs", {}
                                    )
                                    existing = sig_map.get(
                                        "__gemini_function_call_thought_signatures__",
                                        {},
                                    )
                                    if not isinstance(existing, dict):
                                        existing = {}
                                    for tc in next_assistant_msg["tool_calls"]:
                                        tid = tc.get("id")
                                        if tid:
                                            existing[tid] = next_sig_by_id.get(
                                                tid
                                            ) or next_msg_thought_sig
                                    sig_map[
                                        "__gemini_function_call_thought_signatures__"
                                    ] = existing
                                    break
                                elif next_sig_by_id:
                                    sig_map = next_assistant_msg.setdefault(
                                        "additional_kwargs", {}
                                    )
                                    sig_map[
                                        "__gemini_function_call_thought_signatures__"
                                    ] = dict(next_sig_by_id)
                                    break
                                if _next_attempt >= _next_attempts - 1:
                                    raise ValueError(
                                        "Model returned tool calls but no thought signatures; "
                                        "Vertex AI requires thought signatures for function calls. "
                                        "Retries exhausted."
                                    )
                                continue
                            break
                        if next_raw_tc:
                            pass
                        elif _content_breaks_ollama_tool_parse(next_content):
                            next_assistant_msg["content"] = ""
                        _next_recovered_name, _next_recovered_input = (
                            _get_tool_call_from_response(next_response)
                        )
                        if _next_recovered_name is None and next_content:
                            _next_parsed = (
                                strip_json(next_content) if next_content else {}
                            )
                            _next_recovered_name = _next_parsed.get("tool_name")
                            _next_recovered_input = _next_parsed.get("tool_input")
                        if _next_recovered_name is None:
                            _next_recovered_name, _next_recovered_input = (
                                _get_tool_call_from_malformed_response(
                                    next_response
                                )
                            )
                        if not next_raw_tc and _next_recovered_name is not None:
                            if not _did_retry_recovered_next_tool_call:
                                self.messages.append(
                                    {
                                        "role": "user",
                                        "content": "Please use your native tool calling capability to call the tool.",
                                    }
                                )
                                _did_retry_recovered_next_tool_call = True
                                continue
                            next_tool = next(
                                (
                                    t
                                    for t in self.tools
                                    if t.name == _next_recovered_name
                                ),
                                None,
                            )
                            if next_tool is not None:
                                n_message = self._call_tool(
                                    next_tool, _next_recovered_input
                                )
                                if self.debug:
                                    print(
                                        "[DEBUG (agent_base)]: Tool result: ",
                                        n_message,
                                    )
                                if self._model_requires_thought_signatures():
                                    self.messages.append(
                                        {
                                            "role": "user",
                                            "content": f"You called the tool `{_next_recovered_name}`. The tool returned:\n{n_message}\n\nContinue with the next step.",
                                        }
                                    )
                                else:
                                    recovered_id = f"call_{tool_call_id_counter}"
                                    recovered_tc = {
                                        "id": recovered_id,
                                        "name": _next_recovered_name,
                                        "args": _next_recovered_input if isinstance(_next_recovered_input, dict) else ({"input": _next_recovered_input} if _next_recovered_input is not None else {}),
                                        "extra_content": {"google": {"thought_signature": ""}},
                                    }
                                    recovered_assistant_msg = {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [recovered_tc],
                                    }
                                    recovered_assistant_msg.setdefault("additional_kwargs", {})[
                                        "__gemini_function_call_thought_signatures__"
                                    ] = {recovered_id: ""}
                                    self.messages.append(recovered_assistant_msg)
                                    self.messages.append(
                                        {
                                            "role": "tool",
                                            "content": n_message,
                                            "tool_call_id": recovered_id,
                                        }
                                    )
                                tool_call_id_counter += 1
                                continue
                        self.messages.append(next_assistant_msg)
                        next_tool_name, next_tool_input = (
                            _next_recovered_name,
                            _next_recovered_input,
                        )
                        if next_tool_name is None:
                            next_tool_name, next_tool_input = (
                                _get_tool_call_from_response(next_response)
                            )
                        if next_tool_name is None:
                            next_parsed = (
                                strip_json(next_content) if next_content else {}
                            )
                            next_tool_name = next_parsed.get("tool_name")
                            next_tool_input = next_parsed.get("tool_input")
                        if next_tool_name is None:
                            next_tool_name, next_tool_input = (
                                _get_tool_call_from_malformed_response(
                                    next_response
                                )
                            )
                        if next_tool_name is not None:
                            tool = next(
                                (t for t in self.tools if t.name == next_tool_name),
                                None,
                            )
                            if tool is not None:
                                n_message = self._call_tool(tool, next_tool_input)
                                if self.debug:
                                    print(
                                        "[DEBUG (agent_base)]: Tool result: ",
                                        n_message,
                                    )
                                self.messages.append(
                                    {
                                        "role": "tool",
                                        "content": n_message,
                                        "tool_call_id": next_tool_call_id,
                                    }
                                )
                                tool_call_id_counter += 1
                            else:
                                self.messages.append(
                                    {
                                        "role": "user",
                                        "content": f"Tool {next_tool_name} not found. Please try again.",
                                    }
                                )
                                n_message = None
                                break
                        else:
                            n_message = next_parsed.get("content") or next_content
                            break
                    if n_message is None:
                        continue
                    break
                else:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Tool {tool_name} not found. Please try again.",
                        }
                    )
                    n_message = None
                    continue
            else:
                if n_message:
                    parsed = strip_json(n_message)
                    n_message = parsed.get("content") or raw_content
                else:
                    n_message = raw_content
                finish_reason = _get_finish_reason_from_response(_n_message)
                if (
                    (not n_message or not str(n_message).strip())
                    and finish_reason
                    and str(finish_reason).startswith("UNKNOWN_")
                    and not _did_retry_unknown_finish
                ):
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Please continue with your response.",
                        }
                    )
                    _did_retry_unknown_finish = True
                    n_message = None
                    continue
            if self.output_schema is None:
                break
            try:
                parsed = (
                    strip_json(n_message) if isinstance(n_message, str) else n_message
                )
                if (
                    isinstance(n_message, str)
                    and not isinstance(parsed, list)
                    and parsed == {}
                ):
                    try:
                        raw_parsed = json.loads(
                            n_message.strip()
                            .replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                        if isinstance(raw_parsed, list):
                            parsed = raw_parsed
                        elif isinstance(raw_parsed, dict):
                            parsed = raw_parsed
                    except Exception:
                        pass
                    if parsed == {} and "[" in n_message:
                        start = n_message.index("[")
                        depth = 0
                        end = start
                        for i in range(start, len(n_message)):
                            if n_message[i] == "[":
                                depth += 1
                            elif n_message[i] == "]":
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                        if end > start:
                            substring = n_message[start:end]
                            for repair in (_repair_trailing_commas, lambda s: s):
                                try:
                                    raw_parsed = json.loads(repair(substring))
                                    if isinstance(raw_parsed, list):
                                        parsed = raw_parsed
                                        break
                                except Exception:
                                    continue
                    if parsed == {} and "{" in n_message:
                        start = n_message.index("{")
                        try:
                            raw_parsed = json.loads(n_message[start:])
                            if isinstance(raw_parsed, dict) and raw_parsed:
                                parsed = raw_parsed
                        except Exception:
                            pass
                    if parsed == {} and isinstance(n_message, str):
                        raw = (
                            n_message.strip()
                            .replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                        last_obj = _last_json_object(raw)
                        if last_obj:
                            parsed = last_obj
                if isinstance(parsed, list):
                    effective = self._get_effective_output_schema()
                    list_field = _get_single_list_field_name(effective) if effective else None
                    if list_field is not None:
                        parsed = {list_field: parsed}
                if isinstance(parsed, dict) and len(parsed) == 1:
                    effective = self._get_effective_output_schema()
                    list_field = _get_single_list_field_name(effective) if effective else None
                    if list_field is not None:
                        single_key = next(iter(parsed))
                        single_val = parsed[single_key]
                        if isinstance(single_val, list) and single_key != list_field:
                            parsed = {list_field: single_val}
                if parsed:
                    effective = self._get_effective_output_schema()
                    if effective is not None:
                        try:
                            if hasattr(effective, "model_validate"):
                                structured_response = effective.model_validate(parsed)
                            else:
                                structured_response = TypeAdapter(effective).validate_python(
                                    parsed
                                )
                        except ValidationError:
                            list_field = _get_single_list_field_name(effective)
                            if (
                                list_field
                                and isinstance(parsed, dict)
                                and len(parsed) == 1
                                and isinstance(parsed.get(list_field), list)
                            ):
                                fallback = self._validate_list_response_fallback(
                                    effective, list_field, parsed[list_field]
                                )
                                if fallback is not None:
                                    structured_response = fallback
                            if structured_response is None:
                                raise
                break
            except ValidationError as e:
                if self.debug:
                    print("[DEBUG (agent_base)]: ValidationError: ", e)
                print(
                    "[DEBUG (agent_base)]: Error parsing structured response: ",
                    n_message,
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response did not match the required JSON schema. Please try again with the correct format.",
                    }
                )
                n_message = None
                continue
            except Exception as e:
                if self.debug:
                    print("[DEBUG (agent_base)]: ", type(e).__name__, e)
                print(
                    "[DEBUG (agent_base)]: Error parsing structured response: ",
                    n_message,
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response did not match the required JSON schema. Please try again with the correct format.",
                    }
                )
                n_message = None
                continue

        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()
        except ImportError:
            pass

        if (
            isinstance(structured_response, dict)
            and self.output_schema is not None
        ):
            effective = self._get_effective_output_schema()
            if effective is not None:
                try:
                    if hasattr(effective, "model_validate"):
                        structured_response = effective.model_validate(structured_response)
                    else:
                        structured_response = TypeAdapter(effective).validate_python(
                            structured_response
                        )
                except ValidationError:
                    list_field = _get_single_list_field_name(effective)
                    if (
                        list_field
                        and len(structured_response) == 1
                        and isinstance(structured_response.get(list_field), list)
                    ):
                        fallback = self._validate_list_response_fallback(
                            effective, list_field, structured_response[list_field]
                        )
                        if fallback is not None:
                            structured_response = fallback
                except Exception:
                    pass

        return {
            "messages": model_responses,
            "structured_response": structured_response,
        }
