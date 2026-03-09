import json
from typing import Any, Optional

from langchain.tools import BaseTool

from .schema_utils import flatten_json_schema_for_llm, get_output_schema_json_schema


def format_tool_description(tool: BaseTool) -> str:
    args = getattr(tool, "args_schema", None)
    if hasattr(args, "model_json_schema"):
        input_str = json.dumps(args.model_json_schema(), indent=2)
    elif isinstance(args, dict):
        input_str = json.dumps(args, indent=2)
    else:
        input_str = str(args) if args is not None else "N/A"
    return f"""
        {tool.name}: {tool.description}
        Input schema: {input_str}
        """


def build_system_internal_prompt(
    tools: list[BaseTool],
    output_schema: Any,
    model: Any,
    thinking: bool,
) -> str:
    tools_block = ""
    if tools:
        tools_list = "\n".join(format_tool_description(t) for t in tools)
        tools_block = f"""
        You are able to use the following tools:
        {tools_list}

        The input to those tools if required must be ONLY the input specified in the tool's info.

        To call a tool you can use your tool calling caapability or output a JSON object with the following structure:
        ```json
        {{
            "tool_name": "tool_name",
            "tool_input": "tool_input"
        }}
        ```
        {"You MUST NOT use your internal function calling capability, if you want to use a tool just output the json above and nothing else, the system will recall you back to continue the workflow with the output of the tool." if 'gpt' in model and 'oss' in model else ''}
        
        Proceed ONLY one tool at a time, when requesting a tool you must not use more than one tool in the same output, after requesting a tool
        you will be recalled back by the system to continue the workflow.
        """
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
