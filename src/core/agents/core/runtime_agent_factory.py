from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from .agent_base import AgentBase


class RuntimeAgentFactory:
    def __init__(self, create_agent_fn=create_agent, custom_agent_cls=AgentBase):
        self._create_agent_fn = create_agent_fn
        self._custom_agent_cls = custom_agent_cls

    def build(
        self,
        *,
        model: BaseChatModel,
        tools: list[Any],
        system_prompt: str,
        output_schema: Any = None,
        debug: bool = False,
        architecture: str = "custom",
        use_custom_backend: bool = False,
    ):
        if use_custom_backend or architecture == "custom":
            return self._custom_agent_cls(
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                output_schema=output_schema,
                debug=debug,
            )
        return self._create_agent_fn(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            response_format=output_schema,
            debug=debug,
        )


runtime_agent_factory = RuntimeAgentFactory()
