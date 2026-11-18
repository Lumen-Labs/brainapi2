from src.services.mcp.intent_tools import register_intent_tools


def register(context) -> None:
    register_intent_tools(context.mcp, register=context.register_mcp_tool)
