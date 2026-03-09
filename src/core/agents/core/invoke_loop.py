from src.utils.cleanup import strip_json

from .parsing import (
    content_breaks_ollama_tool_parse,
    get_finish_reason_from_response,
    get_first_tool_call_id,
    get_reasoning_from_response,
    get_thought_signature_from_response,
    get_thought_signatures_by_tool_call,
    get_tool_call_from_malformed_response,
    get_tool_call_from_response,
    normalize_invoke_response,
    normalize_message_content,
    serialize_tool_calls,
)
from .prompts import build_system_internal_prompt
from .structured_output import finalize_structured_response, parse_structured_output


def run_invoke_loop(agent, messages, config):
    model_responses = []
    agent.messages = []
    agent.messages.append(
        {
            "role": "system",
            "content": build_system_internal_prompt(
                agent.tools,
                agent.output_schema,
                agent.model,
                agent.thinking,
            )
            + "\n"
            + agent.system_prompt,
        }
    )
    agent.messages.extend(messages)
    tool_call_id_counter = 0

    if agent.debug:
        tags = (config or {}).get("tags", [])
        print(
            "[DEBUG (agent_base)]: Invoking agent with messages: ",
            agent.messages,
            "tags:",
            tags,
        )

    n_message = None
    structured_response = None
    _schema_retry_count = 0
    while True:
        _did_retry_recovered_tool_call = False
        while n_message is None:
            _did_retry_unknown_finish = False
            if agent.output_schema is not None and not agent.tools:
                _n_message = agent.model.invoke(agent.messages, config)
                _n_message = normalize_invoke_response(_n_message)
                model_responses.append(_n_message)
                raw_content = normalize_message_content(
                    _n_message.get("content")
                    if isinstance(_n_message, dict)
                    else getattr(_n_message, "content", None)
                )
                if agent.debug:
                    print("[DEBUG (agent_base)]: Raw content: ", raw_content)
                    reasoning = get_reasoning_from_response(_n_message)
                    if reasoning:
                        print("[DEBUG (agent_base)]: Reasoning: ", reasoning)
                n_message = raw_content
                tool_call_id = (
                    get_first_tool_call_id(_n_message)
                    or f"call_{tool_call_id_counter}"
                )
                assistant_msg = {"role": "assistant", "content": n_message}
                raw_tc = []
            else:
                _invoke_attempts = 3
                for _invoke_attempt in range(_invoke_attempts):
                    _n_message = agent.model.invoke(agent.messages, config)
                    _n_message = normalize_invoke_response(_n_message)
                    model_responses.append(_n_message)
                    raw_content = normalize_message_content(
                        _n_message.get("content")
                        if isinstance(_n_message, dict)
                        else getattr(_n_message, "content", None)
                    )
                    if agent.debug:
                        print("[DEBUG (agent_base)]: Raw content: ", raw_content)
                        reasoning = get_reasoning_from_response(_n_message)
                        if reasoning:
                            print("[DEBUG (agent_base)]: Reasoning: ", reasoning)
                    n_message = raw_content
                    tool_call_id = (
                        get_first_tool_call_id(_n_message)
                        or f"call_{tool_call_id_counter}"
                    )
                    assistant_msg = {
                        "role": "assistant",
                        "content": n_message,
                    }
                    raw_tc = (
                        getattr(_n_message, "tool_calls", None)
                        or (
                            isinstance(_n_message, dict)
                            and _n_message.get("tool_calls")
                        )
                        or []
                    )
                    if not agent.tools and agent.output_schema is not None:
                        raw_tc = []
                    if raw_tc:
                        msg_thought_sig = get_thought_signature_from_response(
                            _n_message
                        )
                        assistant_msg["tool_calls"] = serialize_tool_calls(
                            raw_tc,
                            tool_call_id_counter,
                        )
                        assistant_msg["content"] = ""
                        if not agent._model_requires_thought_signatures():
                            break
                        sig_by_id = get_thought_signatures_by_tool_call(_n_message)
                        if not sig_by_id and msg_thought_sig is not None:
                            for tc in assistant_msg["tool_calls"]:
                                tid = tc.get("id")
                                if tid:
                                    sig_by_id[tid] = msg_thought_sig
                        if sig_by_id:
                            sig_map = assistant_msg.setdefault(
                                "additional_kwargs", {}
                            )
                            existing = sig_map.get(
                                "__gemini_function_call_thought_signatures__", {}
                            )
                            if not isinstance(existing, dict):
                                existing = {}
                            for tc in assistant_msg["tool_calls"]:
                                tid = tc.get("id")
                                if tid:
                                    existing[tid] = (
                                        sig_by_id.get(tid) or msg_thought_sig
                                    )
                            sig_map[
                                "__gemini_function_call_thought_signatures__"
                            ] = existing
                            break
                        sig_map = assistant_msg.setdefault("additional_kwargs", {})
                        existing = sig_map.get(
                            "__gemini_function_call_thought_signatures__", {}
                        )
                        if not isinstance(existing, dict):
                            existing = {}
                        for tc in assistant_msg["tool_calls"]:
                            tid = tc.get("id")
                            if tid:
                                existing[tid] = ""
                        sig_map["__gemini_function_call_thought_signatures__"] = (
                            existing
                        )
                        break
        if raw_tc:
            pass
        elif content_breaks_ollama_tool_parse(n_message):
            assistant_msg["content"] = ""
        _recovered_name, _recovered_input = get_tool_call_from_response(_n_message)
        if _recovered_name is None and n_message:
            _parsed = strip_json(n_message)
            _recovered_name = _parsed.get("tool_name")
            _recovered_input = _parsed.get("tool_input")
        if _recovered_name is None:
            _recovered_name, _recovered_input = (
                get_tool_call_from_malformed_response(_n_message)
            )
        if not raw_tc and _recovered_name is not None:
            if not _did_retry_recovered_tool_call:
                agent.messages.append(
                    {
                        "role": "user",
                        "content": "Please use your native tool calling capability to call the tool.",
                    }
                )
                _did_retry_recovered_tool_call = True
                n_message = None
                continue
            tool = next(
                (t for t in agent.tools if t.name == _recovered_name),
                None,
            )
            if tool is not None:
                n_message = agent._call_tool(tool, _recovered_input)
                if agent._model_requires_thought_signatures():
                    agent.messages.append(
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
                    agent.messages.append(recovered_assistant_msg)
                    agent.messages.append(
                        {
                            "role": "tool",
                            "content": n_message,
                            "tool_call_id": recovered_id,
                        }
                    )
                tool_call_id_counter += 1
                n_message = None
                continue
        agent.messages.append(assistant_msg)
        tool_name, tool_input = _recovered_name, _recovered_input
        if tool_name is None:
            tool_name, tool_input = get_tool_call_from_response(_n_message)
        if tool_name is None and n_message:
            parsed = strip_json(n_message)
            tool_name = parsed.get("tool_name")
            tool_input = parsed.get("tool_input")
        if tool_name is None:
            tool_name, tool_input = get_tool_call_from_malformed_response(
                _n_message
            )
        if tool_name is not None:
            tool = next(
                (t for t in agent.tools if t.name == tool_name),
                None,
            )
            if tool is not None:
                n_message = agent._call_tool(tool, tool_input)
                agent.messages.append(
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
                        next_response = agent.model.invoke(agent.messages, config)
                        next_response = normalize_invoke_response(next_response)
                        if agent.debug:
                            print(
                                "[DEBUG (agent_base)]: Next response: ",
                                next_response,
                            )
                            reasoning = get_reasoning_from_response(next_response)
                            if reasoning:
                                print(
                                    "[DEBUG (agent_base)]: Reasoning: ",
                                    reasoning,
                                )
                        model_responses.append(next_response)
                        next_content = normalize_message_content(
                            next_response.get("content")
                            if isinstance(next_response, dict)
                            else getattr(next_response, "content", None)
                        )
                        next_tool_call_id = (
                            get_first_tool_call_id(next_response)
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
                                get_thought_signature_from_response(next_response)
                            )
                            next_assistant_msg["tool_calls"] = serialize_tool_calls(
                                next_raw_tc,
                                tool_call_id_counter,
                            )
                            next_assistant_msg["content"] = ""
                            if not agent._model_requires_thought_signatures():
                                break
                            next_sig_by_id = get_thought_signatures_by_tool_call(
                                next_response
                            )
                            if (
                                not next_sig_by_id
                                and next_msg_thought_sig is not None
                            ):
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
                                        existing[tid] = (
                                            next_sig_by_id.get(tid)
                                            or next_msg_thought_sig
                                        )
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
                                    existing[tid] = ""
                            sig_map[
                                "__gemini_function_call_thought_signatures__"
                            ] = existing
                            break
                        break
                    if next_raw_tc:
                        pass
                    elif content_breaks_ollama_tool_parse(next_content):
                        next_assistant_msg["content"] = ""
                    _next_recovered_name, _next_recovered_input = (
                        get_tool_call_from_response(next_response)
                    )
                    if _next_recovered_name is None and next_content:
                        _next_parsed = (
                            strip_json(next_content) if next_content else {}
                        )
                        _next_recovered_name = _next_parsed.get("tool_name")
                        _next_recovered_input = _next_parsed.get("tool_input")
                    if _next_recovered_name is None:
                        _next_recovered_name, _next_recovered_input = (
                            get_tool_call_from_malformed_response(next_response)
                        )
                    if not next_raw_tc and _next_recovered_name is not None:
                        if not _did_retry_recovered_next_tool_call:
                            agent.messages.append(
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
                                for t in agent.tools
                                if t.name == _next_recovered_name
                            ),
                            None,
                        )
                        if next_tool is not None:
                            n_message = agent._call_tool(
                                next_tool, _next_recovered_input
                            )
                            if agent.debug:
                                print(
                                    "[DEBUG (agent_base)]: Tool result: ",
                                    n_message,
                                )
                            if agent._model_requires_thought_signatures():
                                agent.messages.append(
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
                                    "args": (
                                        _next_recovered_input
                                        if isinstance(_next_recovered_input, dict)
                                        else (
                                            {"input": _next_recovered_input}
                                            if _next_recovered_input is not None
                                            else {}
                                        )
                                    ),
                                    "extra_content": {
                                        "google": {"thought_signature": ""}
                                    },
                                }
                                recovered_assistant_msg = {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [recovered_tc],
                                }
                                recovered_assistant_msg.setdefault(
                                    "additional_kwargs", {}
                                )["__gemini_function_call_thought_signatures__"] = {
                                    recovered_id: ""
                                }
                                agent.messages.append(recovered_assistant_msg)
                                agent.messages.append(
                                    {
                                        "role": "tool",
                                        "content": n_message,
                                        "tool_call_id": recovered_id,
                                    }
                                )
                            tool_call_id_counter += 1
                            continue
                    agent.messages.append(next_assistant_msg)
                    next_tool_name, next_tool_input = (
                        _next_recovered_name,
                        _next_recovered_input,
                    )
                    if next_tool_name is None:
                        next_tool_name, next_tool_input = (
                            get_tool_call_from_response(next_response)
                        )
                    if next_tool_name is None:
                        next_parsed = (
                            strip_json(next_content) if next_content else {}
                        )
                        next_tool_name = next_parsed.get("tool_name")
                        next_tool_input = next_parsed.get("tool_input")
                    if next_tool_name is None:
                        next_tool_name, next_tool_input = (
                            get_tool_call_from_malformed_response(next_response)
                        )
                    if next_tool_name is not None:
                        tool = next(
                            (t for t in agent.tools if t.name == next_tool_name),
                            None,
                        )
                        if tool is not None:
                            n_message = agent._call_tool(tool, next_tool_input)
                            if agent.debug:
                                print(
                                    "[DEBUG (agent_base)]: Tool result: ",
                                    n_message,
                                )
                            agent.messages.append(
                                {
                                    "role": "tool",
                                    "content": n_message,
                                    "tool_call_id": next_tool_call_id,
                                }
                            )
                            tool_call_id_counter += 1
                        else:
                            agent.messages.append(
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
                agent.messages.append(
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
            finish_reason = get_finish_reason_from_response(_n_message)
            if (
                (not n_message or not str(n_message).strip())
                and finish_reason
                and str(finish_reason).startswith("UNKNOWN_")
                and not _did_retry_unknown_finish
            ):
                agent.messages.append(
                    {
                        "role": "user",
                        "content": "Please continue with your response.",
                    }
                )
                _did_retry_unknown_finish = True
                n_message = None
                continue
        if agent.output_schema is None:
            break
        try:
            structured_response = parse_structured_output(
                n_message, agent._get_effective_output_schema
            )
            break
        except Exception as e:
            if agent.debug:
                print("[DEBUG (agent_base)]: ", type(e).__name__, e)
            print(
                "[DEBUG (agent_base)]: Error parsing structured response: ",
                n_message,
            )
            if (
                agent.output_schema is not None
                and not agent.tools
                and _schema_retry_count >= 1
            ):
                break
            agent.messages.append(
                {
                    "role": "user",
                    "content": "Your previous response did not match the required JSON schema. Please try again with the correct format.",
                }
            )
            _schema_retry_count += 1
            n_message = None
            continue

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except ImportError:
        pass

    structured_response = finalize_structured_response(
        structured_response,
        agent.output_schema,
        agent._get_effective_output_schema,
    )

    return {
        "messages": model_responses,
        "structured_response": structured_response,
    }
