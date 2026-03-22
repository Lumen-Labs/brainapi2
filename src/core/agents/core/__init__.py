"""
File: /__init__.py
Project: core
Created Date: Wednesday February 25th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday March 4th 2026 9:35:41 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

try:
    from .agent_base import AgentBase, parse_structured_from_messages
except ModuleNotFoundError:
    AgentBase = None

    def parse_structured_from_messages(*args, **kwargs):
        from .agent_base import parse_structured_from_messages as _delegate

        return _delegate(*args, **kwargs)


from .runtime_agent_factory import RuntimeAgentFactory, runtime_agent_factory

__all__ = [
    "AgentBase",
    "parse_structured_from_messages",
    "RuntimeAgentFactory",
    "runtime_agent_factory",
]
