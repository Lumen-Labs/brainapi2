class McpToolError(Exception):
    """Raised from MCP tools so FastMCP returns isError=true."""


def raise_tool_error(message: str) -> None:
    raise McpToolError(message)


def raise_if_tool_error_result(result):
    if isinstance(result, str) and (
        result in {"Unauthorized", "Fact not found"} or result.startswith("Error")
    ):
        raise McpToolError(result)
    return result
