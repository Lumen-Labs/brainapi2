from typing import Any

def _default_create_agent(**kwargs):
    from langchain.agents import create_agent

    return create_agent(**kwargs)


class RuntimeAgentFactory:
    def __init__(self, create_agent_fn=None, custom_agent_cls=None):
        self._create_agent_fn = create_agent_fn or _default_create_agent
        self._custom_agent_cls = custom_agent_cls

    def build(
        self,
        *,
        model: Any,
        tools: list[Any],
        system_prompt: str,
        output_schema: Any = None,
        debug: bool = False,
        architecture: str = "custom",
        use_custom_backend: bool = False,
    ):
        if use_custom_backend or architecture == "custom":
            custom_agent_cls = self._custom_agent_cls
            if custom_agent_cls is None:
                from .agent_base import AgentBase

                custom_agent_cls = AgentBase
            return custom_agent_cls(
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
