import os
import warnings

import langsmith
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .invoke_loop import run_invoke_loop
from .parsing import parse_structured_from_messages
from .schema_utils import get_effective_output_schema
from .types import AgentMessage, MessagesDict

warnings.filterwarnings(
    "ignore",
    message=r"Unrecognized FinishReason enum value",
    category=UserWarning,
)


class AgentBase:
    model: BaseChatModel
    system_prompt: str
    tools: list[BaseTool]
    output_schema: BaseModel | None
    debug: bool
    thinking: bool
    messages: list[AgentMessage]

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        tools: list[BaseTool],
        output_schema: BaseModel | None = None,
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
        return get_effective_output_schema(self.output_schema)

    def _normalize_tool_input(self, tool: BaseTool, raw_input) -> str | dict:
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

    def _call_tool(self, tool: BaseTool, input):
        normalized = self._normalize_tool_input(tool, input)
        if self.debug:
            print(
                f"[DEBUG (agent_base)]: calling tool {tool.name} with input: {normalized}"
            )
        return tool.run(normalized)

    def invoke(self, messages: MessagesDict, config: dict | None = None):
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
            return run_invoke_loop(self, messages.get("messages"), config)
