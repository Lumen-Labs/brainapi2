"""Invocation limits scoped to one free-text enrichment operation."""

import json
from contextvars import ContextVar

# None means no enrichment scope. A tuple also represents the default output cap.
invocation_limits = ContextVar("ingestion_invocation_limits", default=None)
source_span_tokens = ContextVar("ingestion_source_span_tokens", default=0)
scout_payload_size = ContextVar("ingestion_scout_payload_size", default=(0, 0))


def estimated_relationship_output(source_tokens, entity_count, entity_tokens):
    """Estimate response growth from source evidence and potential entity pairs."""
    pairs = entity_count * max(0, entity_count - 1) // 2
    return max(4 * source_tokens, 2 * entity_tokens + 8 * pairs)


# Reserve more generated tokens for relationship extraction than entity discovery.
# These ratios are estimates. Dense spans can still need a smaller retry.
_SCOUT_OUTPUT_GROWTH = 2



class IngestModelLimitError(RuntimeError):
    """A model request or response exceeded its supported size."""


def is_size_error(error):
    if isinstance(error, IngestModelLimitError) or type(error).__name__ == "ArchitectOutputLimitError":
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "context_length_exceeded", "maximum context length", "context window exceeded",
        "input is too long", "prompt is too long", "input token count exceeds",
        "input is too long for requested model", "too many input tokens",
    ))


def _positive_limit(value):
    return value if isinstance(value, int) and value > 0 else None


def _profile_limit(model, key):
    profile = getattr(model, "profile", None)
    return _positive_limit(profile.get(key)) if isinstance(profile, dict) else None


def limit_output(model):
    limits = invocation_limits.get()
    if not limits or limits[0] is None:
        return model
    cap = limits[0]
    model_cap = _profile_limit(model, "max_output_tokens")
    if model_cap:
        cap = min(cap, model_cap)
    fields = getattr(type(model), "model_fields", {})
    if "model_kwargs" in fields and any(
        "bedrock" in cls.__module__.lower() for cls in type(model).__mro__
    ):
        return model.model_copy(update={
            "max_tokens": cap,
            "model_kwargs": {**(model.model_kwargs or {}), "max_tokens": cap},
        })
    for field in ("max_tokens", "max_output_tokens", "num_predict", "max_completion_tokens"):
        if field in fields:
            return model.model_copy(update={field: cap})
    return model


def check_input(model, messages, tools=(), schema=None):
    if invocation_limits.get() is None:
        return
    from src.core.saving.ingest_cost import count_source_tokens

    from src.core.saving.pipeline_runtime import current_stage

    context = _profile_limit(model, "max_input_tokens") or _positive_limit(
        getattr(model, "num_ctx", None)
    )
    model_output = _profile_limit(model, "max_output_tokens")
    output = next((n for n in (
        getattr(model, "max_tokens", None),
        getattr(model, "max_output_tokens", None),
        getattr(model, "num_predict", None),
        getattr(model, "max_completion_tokens", None),
        (getattr(model, "model_kwargs", None) or {}).get("max_tokens"),
    ) if _positive_limit(n)), model_output or 4096)
    if model_output:
        output = min(output, model_output)
    input_limit = context - output if context else 24576
    stage = current_stage()
    source_tokens = source_span_tokens.get()
    estimate = 0
    if stage == "scout":
        estimate = _SCOUT_OUTPUT_GROWTH * source_tokens
    elif stage == "architect":
        estimate = estimated_relationship_output(source_tokens, *scout_payload_size.get())
    if estimate > output:
        raise IngestModelLimitError("Source span exceeds the estimated extraction capacity")

    schemas = []
    for tool in tools:
        spec = getattr(tool, "args_schema", None)
        schemas.append((getattr(tool, "name", ""), getattr(tool, "description", ""), spec.model_json_schema() if hasattr(spec, "model_json_schema") else spec))
    if schema is not None and hasattr(schema, "model_json_schema"):
        schemas.append(schema.model_json_schema())
    overhead = json.dumps(schemas, ensure_ascii=False, default=str) if schemas else ""
    counter = getattr(model, "get_num_tokens_from_messages", None)
    # The generic LangChain counter may download a GPT-2 tokenizer. Use
    # adapter-specific counters when the adapter actually overrides it.
    if callable(counter) and "langchain_core.language_models" not in getattr(counter, "__module__", ""):
        try:
            used = counter(messages) + count_source_tokens(overhead)[0]
        except (NotImplementedError, ImportError, ValueError, TypeError):
            used = None
    else:
        used = None
    if used is None:
        payload = json.dumps(messages, ensure_ascii=False, default=str) + overhead
        used = (count_source_tokens(payload)[0] * 5 + 3) // 4
    if used + 128 > input_limit:
        raise IngestModelLimitError("Assembled input exceeds the model context limit")


def check_output(response):
    if invocation_limits.get() is None:
        return response
    metadata = (response.get("response_metadata") if isinstance(response, dict)
                else getattr(response, "response_metadata", None)) or {}
    additional = (response.get("additional_kwargs") if isinstance(response, dict)
                  else getattr(response, "additional_kwargs", None)) or {}
    reason = metadata.get("stopReason") or metadata.get("stop_reason") or metadata.get("finish_reason") or additional.get("finish_reason")
    if str(reason).lower() in {"length", "max_tokens", "max_output_tokens", "model_length", "max_tokens_exceeded"}:
        raise IngestModelLimitError("Model response exceeded the output limit")
    return response


def invoke_enrichment_model(agent, config):
    if invocation_limits.get() is not None:
        check_input(agent._base_model, agent.messages, agent.tools if agent._tools_bound else ())
    return agent.model.invoke(agent.messages, config)


def wrap_enrichment_model_call(request, handler):
    messages = [request.system_message, *request.messages] if request.system_message else request.messages
    check_input(request.model, messages, request.tools, getattr(request.response_format, "schema", None))
    response = handler(request)
    for message in response.result:
        check_output(message)
    return response
