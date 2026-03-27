import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path: str) -> ast.Module:
    return ast.parse(read_source(relative_path))


def class_names(relative_path: str) -> set[str]:
    tree = parse_source(relative_path)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def method_names(relative_path: str, class_name: str) -> set[str]:
    tree = parse_source(relative_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                method.name
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_name} not found in {relative_path}")


class KGAgentPromptStrategyArchitectureTests(unittest.TestCase):
    def test_kg_agent_has_prompt_registry_mapping(self):
        source = read_source("src/core/agents/kg_agent.py")
        self.assertIn("_SYSTEM_PROMPT_BUILDERS", source)
        self.assertIn('"normal"', source)
        self.assertIn('"graph-consolidator"', source)

    def test_kg_agent_validates_prompt_type_from_registry(self):
        source = read_source("src/core/agents/kg_agent.py")
        self.assertIn("if type_ not in self._SYSTEM_PROMPT_BUILDERS:", source)
        self.assertIn('raise ValueError(f"Invalid type: {type_}")', source)
        self.assertIn("system_prompt = self._SYSTEM_PROMPT_BUILDERS[type_](", source)


class JanitorAgentPromptAndSchemaArchitectureTests(unittest.TestCase):
    def test_janitor_agent_has_prompt_registry_mapping(self):
        source = read_source("src/core/agents/janitor_agent.py")
        self.assertIn("_SYSTEM_PROMPT_BUILDERS", source)
        self.assertIn('"janitor"', source)
        self.assertIn('"graph-janitor"', source)
        self.assertIn('"atomic-janitor"', source)

    def test_janitor_agent_uses_schema_resolution_methods(self):
        methods = method_names("src/core/agents/janitor_agent.py", "JanitorAgent")
        self.assertIn("_resolve_response_format", methods)
        self.assertIn("_normalize_schema", methods)
        source = read_source("src/core/agents/janitor_agent.py")
        self.assertIn(
            "response_format = self._resolve_response_format(output_schema, output_schemas)",
            source,
        )

    def test_janitor_agent_validates_prompt_type(self):
        source = read_source("src/core/agents/janitor_agent.py")
        self.assertIn("if type_ not in self._SYSTEM_PROMPT_BUILDERS:", source)
        self.assertIn('raise ValueError(f"Invalid type: {type_}")', source)


class ArchitectAgentPromptStrategyArchitectureTests(unittest.TestCase):
    def test_architect_agent_has_prompt_registry_mapping(self):
        source = read_source("src/core/agents/architect_agent.py")
        self.assertIn("_SYSTEM_PROMPT_BUILDERS", source)
        self.assertIn('("single", "granular")', source)
        self.assertIn('("single", "coarse")', source)
        self.assertIn('("tooler", "granular")', source)
        self.assertIn('("tooler", "coarse")', source)

    def test_architect_agent_uses_prompt_resolver(self):
        methods = method_names("src/core/agents/architect_agent.py", "ArchitectAgent")
        self.assertIn("_resolve_system_prompt", methods)
        source = read_source("src/core/agents/architect_agent.py")
        self.assertIn(
            "system_prompt = self._resolve_system_prompt(type_, mode, extra_system_prompt)",
            source,
        )
        self.assertIn('raise ValueError(f"Invalid mode for architect agent: {mode}")', source)
        self.assertIn('raise ValueError(f"Invalid type for architect agent: {type_}")', source)


class JanitorToolReuseArchitectureTests(unittest.TestCase):
    def test_execute_graph_operation_tool_is_alias_subclass(self):
        source = read_source(
            "src/core/agents/tools/janitor_agent/JanitorAgentExecuteGraphOperationTool.py"
        )
        self.assertIn(
            "class JanitorAgentExecuteGraphOperationTool(JanitorAgentExecuteGraphReadOperationTool):",
            source,
        )
        self.assertIn("pass", source)
        self.assertNotIn("def _run", source)

    def test_janitor_tool_classes_exist(self):
        names = class_names(
            "src/core/agents/tools/janitor_agent/JanitorAgentExecuteGraphReadOperationTool.py"
        )
        self.assertIn("JanitorAgentExecuteGraphReadOperationTool", names)
        names_alias = class_names(
            "src/core/agents/tools/janitor_agent/JanitorAgentExecuteGraphOperationTool.py"
        )
        self.assertIn("JanitorAgentExecuteGraphOperationTool", names_alias)


if __name__ == "__main__":
    unittest.main()
