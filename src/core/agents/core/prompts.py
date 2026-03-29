import json
from typing import Any, Optional

from langchain.tools import BaseTool

from .schema_utils import flatten_json_schema_for_llm, get_output_schema_json_schema


def _tool_schema_str(tool: BaseTool) -> str:
    args = getattr(tool, "args_schema", None)
    if hasattr(args, "model_json_schema"):
        return json.dumps(args.model_json_schema())
    if isinstance(args, dict):
        return json.dumps(args)
    return str(args) if args is not None else ""


def format_tool_description(tool: BaseTool, include_schema: bool = False) -> str:
    base = f"- {tool.name}: {tool.description}"
    if include_schema:
        schema = _tool_schema_str(tool)
        if schema:
            return f"{base} Schema: {schema}"
    return base


def build_system_internal_prompt(
    tools: list[BaseTool],
    output_schema: Any,
    model: Any,
    thinking: bool,
    tools_bound: bool = True,
) -> str:
    tools_block = ""
    if tools:
        tools_list = "\n".join(
            format_tool_description(t, include_schema=not tools_bound) for t in tools
        )
        _oss = "gpt" in str(model) and "oss" in str(model)
        _json_fallback = (
            'If you cannot use native tool calling, output ONLY: {"tool_name": "...", "tool_input": {...}} and nothing else.'
            if _oss
            else ""
        )
        tools_block = f"""Tools:\n{tools_list}\n\nCall one tool at a time. {_json_fallback}"""
    output_schema_block = ""
    if output_schema:
        schema_dict = get_output_schema_json_schema(output_schema)
        if schema_dict is not None:
            output_schema_block = f"""
        Your output must be a JSON object ONLY with the following JSON schema structure:
        {json.dumps(flatten_json_schema_for_llm(schema_dict), indent=2)}
        You must strictly follow the schema since it's a JSON Schema Specification, don't add any additional fields or properties.
        Don't output the same fields of the JSON schema as it is, it's just the schema instructions, you must output the actual data.
        Remember that the fields that you are interested in are inside the 'properties' key of the JSON schema.
        You must return the JSON ONLY, no additional text or comments or explanation.
        """
    return f"""
        You are a helpful agent. You must follow the instructions given to you by the user strictly.
        {tools_block}
        {output_schema_block}
        {"/no_think" if not thinking else ""}
        """
