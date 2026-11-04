"""Operation-scoped routing and durable per-call usage; never mutate process defaults."""
from contextvars import ContextVar
from functools import lru_cache
from uuid import uuid4

_context = ContextVar("pipeline_operation", default=None)
_stage = ContextVar("pipeline_stage_name", default=None)


def bind_operation(value):
    return _context.set(value)


def reset_operation(token):
    _context.reset(token)


@lru_cache(maxsize=32)
def route_client(provider, model, region):
    from src.config import config
    if provider == "amazon_bedrock":
        if not config.bedrock:
            raise ValueError("Bedrock credentials are not configured on this VM")
        if region and region != config.bedrock.region:
            raise ValueError("Routing region must match the VM's configured region")
        from src.lib.llm.client_bedrock import _BedrockLLMBase
        return _BedrockLLMBase(model)
    if provider == "anthropic_vertex":
        if not config.gcp:
            raise ValueError("Vertex credentials are not configured on this VM")
        from src.lib.llm.client_anthropic_vertex import _AnthropicVertexLLMBase, _vertex_region
        if region and region != _vertex_region():
            raise ValueError("Routing region must match the VM's configured region")
        return _AnthropicVertexLLMBase(model)
    raise ValueError("Provider is not supported for stage overrides")


def routed_client(fallback, role="small"):
    context = _context.get() or {}
    stage = _stage.get()
    for route in (context.get("configuration") or {}).get("routes", []):
        if route["stage"] == f"ingestion.{stage}":
            return route_client(route["provider"], route["model"], route.get("region"))
    selected = (context.get("configuration") or {}).get("defaults", {}).get(role)
    if selected:
        current_provider, current_model = configured_model(role)
        if selected["provider"] != current_provider or selected["model"] != current_model:
            return route_client(selected["provider"], selected["model"], None)
    return fallback


def configured_model(role="small"):
    from src.config import config
    provider = config.llm_large_provider if role == "large" else config.llm_small_provider
    if provider == "ollama":
        return provider, getattr(config, f"llm_{role}_model", None) or "unknown"
    source = {"amazon_bedrock":"bedrock","anthropic_vertex":"gcp","gcp_vertex":"gcp","openai":"openai","azure":"azure","anthropic":"anthropic","deepseek":"deepseek"}.get(provider)
    settings = getattr(config, source, None) if source else None
    return provider, getattr(settings, f"{role}_llm_model", None) or "unknown"


def model_identity():
    from src.config import config
    context = _context.get() or {}
    stage = _stage.get()
    for route in (context.get("configuration") or {}).get("routes", []):
        if route["stage"] == f"ingestion.{stage}":
            return route["provider"], route["model"]
    return configured_model()


def record_call(response, error=None, identity=None, quality=None):
    context = _context.get()
    stage = _stage.get()
    if not context or not stage:
        return
    from src.services.data.main import data_adapter
    usage = getattr(response, "usage_metadata", None)
    if isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usage") or usage
    usage = usage or {}
    provider, model = identity or model_identity()
    details = usage.get("input_token_details") or {}
    event = {"_id":str(uuid4()), "operation_id":context["import_id"],
        "section_id":context.get("section_id"), "page_id":context.get("page_id"), "task_id":context["task_id"], "attempt":context.get("attempt", 1),
        "stage":stage, "provider":provider, "model":model,
        "input_tokens":usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens":usage.get("output_tokens", usage.get("completion_tokens")),
        "cache_read_tokens":details.get("cache_read", usage.get("cache_read_input_tokens", 0)),
        "cache_write_tokens":details.get("cache_creation", usage.get("cache_write_input_tokens", usage.get("cache_creation_input_tokens", 0))),
        "quality":quality or ("reported" if usage else "unavailable"), "outcome":"error" if error else "completed",
        "configuration_version":(context.get("configuration") or {}).get("version", "legacy")}
    from src.core.saving.pipeline_store import collection
    collection(context["brain_id"], "pipeline_usage_events").insert_one(event)


def instrument_model(model):
    if not _context.get():
        return model
    from langchain_core.callbacks import BaseCallbackHandler
    fallback_provider, fallback_model = model_identity()
    actual_model = getattr(model,"model_id",None) or getattr(model,"model_name",None) or getattr(model,"model",None) or fallback_model
    module = type(model).__module__.lower()
    actual_provider = "amazon_bedrock" if "bedrock" in module else "anthropic_vertex" if "vertex" in module and "anthropic" in type(model).__name__.lower() else "gcp_vertex" if "vertex" in module else "azure" if "azure" in module else "openai" if "openai" in module else fallback_provider
    identity = (actual_provider, actual_model)
    class UsageCallback(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            generation = response.generations[0][0] if response.generations and response.generations[0] else None
            message = getattr(generation, "message", None)
            record_call(message or {"usage":(response.llm_output or {}).get("usage") or (response.llm_output or {}).get("token_usage")}, identity=identity)
        def on_llm_error(self, error, **kwargs):
            record_call(None, error, identity=identity)
    return model.model_copy(update={"callbacks":[*(model.callbacks or []), UsageCallback()]})


def record_embedding(client, texts, error=None):
    if not _context.get():
        return
    from src.config import config
    from src.core.saving.ingest_cost import count_source_tokens
    token = _stage.set("embed")
    try:
        record_call({"usage_metadata":{"input_tokens":sum(count_source_tokens(text)[0] for text in texts),"output_tokens":0}} if not error else None,
            error=error, identity=(config.embeddings_provider, getattr(client,"model","unknown")), quality="estimated" if not error else "unavailable")
    finally:
        _stage.reset(token)


def routing_stage(name):
    from functools import wraps
    def decorate(fn):
        @wraps(fn)
        def run(*args, **kwargs):
            current = _stage.get()
            token = _stage.set("consolidation" if current == "consolidation" else name)
            context = _context.get()
            if context and context.get("section_id"):
                from src.core.saving.ingest_manifest import update_section
                update_section(context["brain_id"],context["task_id"],"started",stage=_stage.get())
            try:
                return fn(*args, **kwargs)
            finally:
                _stage.reset(token)
        return run
    return decorate


def child_pipeline_headers(task, child_task_id):
    headers = getattr(task.request,"headers",None)
    if not isinstance(headers,dict) or not headers.get("pipeline_operation"):
        return {}
    context = {**headers["pipeline_operation"], "task_id":child_task_id,"section_id":child_task_id,"page_id":child_task_id}
    return {"headers":{"pipeline_operation":context}}


def record_cleanup_usage(cleaned, url):
    context = _context.get()
    if not context or not any(key in cleaned for key in ("model","input_tokens","output_tokens")):
        return
    import hashlib
    token = _stage.set("cleanup")
    operation = bind_operation({**context,"page_id":hashlib.sha256(url.encode()).hexdigest()[:24],"section_id":None})
    try:
        usage = {"input_tokens":cleaned.get("input_tokens"),"output_tokens":cleaned.get("output_tokens")}
        quality = "reported" if all(isinstance(v,int) and v>=0 for v in usage.values()) else "unavailable"
        record_call({"usage_metadata":usage},identity=(cleaned.get("provider") or "lumen_crawler",cleaned.get("model") or "unknown"),quality=quality)
    finally:
        reset_operation(operation)
        _stage.reset(token)
