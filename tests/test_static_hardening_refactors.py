import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path: str) -> ast.Module:
    return ast.parse(read_source(relative_path))


def get_function_default(
    relative_path: str, function_name: str, arg_name: str
) -> ast.AST | None:
    tree = parse_source(relative_path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != function_name:
                continue
            args = node.args.args
            defaults = node.args.defaults
            default_offset = len(args) - len(defaults)
            for index, arg in enumerate(args):
                if arg.arg == arg_name and index >= default_offset:
                    return defaults[index - default_offset]
            return None
    raise AssertionError(f"Function {function_name} not found in {relative_path}")


class MutableDefaultGuardTests(unittest.TestCase):
    def test_graph_update_properties_uses_none_defaults(self):
        new_properties_default = get_function_default(
            "src/adapters/graph.py", "update_properties", "new_properties"
        )
        properties_to_remove_default = get_function_default(
            "src/adapters/graph.py", "update_properties", "properties_to_remove"
        )
        self.assertIsInstance(new_properties_default, ast.Constant)
        self.assertIsNone(new_properties_default.value)
        self.assertIsInstance(properties_to_remove_default, ast.Constant)
        self.assertIsNone(properties_to_remove_default.value)

    def test_cleanup_strip_properties_uses_none_default(self):
        pop_also_default = get_function_default(
            "src/utils/cleanup.py", "strip_properties", "pop_also"
        )
        self.assertIsInstance(pop_also_default, ast.Constant)
        self.assertIsNone(pop_also_default.value)

    def test_entities_get_entity_status_uses_none_default(self):
        types_default = get_function_default(
            "src/services/api/controllers/entities.py", "get_entity_status", "types"
        )
        self.assertIsInstance(types_default, ast.Constant)
        self.assertIsNone(types_default.value)

    def test_routes_get_entity_status_uses_none_default(self):
        types_default = get_function_default(
            "src/services/api/routes/retrieve.py", "get_entity_status", "types"
        )
        self.assertIsInstance(types_default, ast.Constant)
        self.assertIsNone(types_default.value)


class GraphReductionArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.source = read_source("src/adapters/graph.py")
        self.tree = parse_source("src/adapters/graph.py")

    def test_strategy_classes_exist(self):
        class_names = {
            node.name for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
        }
        self.assertIn("NeighborVectorReductionStrategy", class_names)
        self.assertIn("SimilarityOnlyReductionStrategy", class_names)
        self.assertIn("DescriptionAwareReductionStrategy", class_names)
        self.assertIn("NeighborVectorReductionStrategyFactory", class_names)

    def test_graph_adapter_init_accepts_reduction_factory(self):
        self.assertIn(
            "reduction_strategy_factory: Optional[NeighborVectorReductionStrategyFactory] = None",
            self.source,
        )
        self.assertIn(
            "self._reduction_strategy_factory =",
            self.source,
        )

    def test_reduce_neighbor_vectors_uses_strategy_factory(self):
        self.assertIn("if not vectors_with_desc or not averaged_vector:", self.source)
        self.assertIn(
            "strategy = self._reduction_strategy_factory.create(description)",
            self.source,
        )
        self.assertIn(
            "or len(embeddings) != len(averaged_vector)",
            self.source,
        )

    def test_get_2nd_degree_hops_uses_averaging_helper(self):
        self.assertIn("averaged_vector = self._average_embeddings(vs)", self.source)
        self.assertIn("if len(averaged_vector) == 0:", self.source)
        self.assertIn("all_filtered_fd_uuids: set[str] = set()", self.source)


class EntityStatusHardeningTests(unittest.TestCase):
    def test_entity_status_skips_missing_graph_rows(self):
        source = read_source("src/services/api/controllers/entities.py")
        self.assertIn(
            "nodes = graph_adapter.get_by_uuids([target_node_id], brain_id=brain_id)",
            source,
        )
        self.assertIn("if len(nodes) == 0:", source)
        self.assertIn("continue", source)


class VectorSearchFacadeArchitectureTests(unittest.TestCase):
    def test_vector_search_facade_file_exists_with_store_helpers(self):
        source = read_source("src/utils/vector_search.py")
        self.assertIn("class VectorSearchFacade", source)
        self.assertIn("def search_nodes(", source)
        self.assertIn("def search_triplets(", source)
        self.assertIn("def search_relationships(", source)
        self.assertIn("def search_data(", source)
        self.assertIn("store=store", source)
        self.assertIn("brain_id=brain_id", source)

    def test_retrieve_data_uses_facade_methods_not_positional_calls(self):
        source = read_source("src/services/api/controllers/retrieve.py")
        self.assertIn("vector_search = VectorSearchFacade(vector_store_adapter)", source)
        self.assertIn("vector_search.search_data(", source)
        self.assertIn("vector_search.search_triplets(", source)
        self.assertIn("k=limit", source)
        self.assertNotIn(
            'vector_store_adapter.search_vectors(\n            text_embeddings.embeddings, "data", brain_id, limit',
            source,
        )
        self.assertNotIn(
            'vector_store_adapter.search_vectors(\n            text_embeddings.embeddings, "triplets", brain_id, limit',
            source,
        )

    def test_key_modules_use_facade_or_keyword_safe_search(self):
        expected_snippets = {
            "src/services/api/controllers/entities.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/services/api/controllers/kg.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/services/mcp/main.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/core/search/entity_context.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/core/search/entity_info.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/core/search/entity_sibilings.py": [
                "VectorSearchFacade",
                "vector_search.search_nodes(",
            ],
            "src/core/agents/tools/kg_agent/KGAgentSearchGraphTool.py": [
                'store="triplets"',
                'store="nodes"',
                "brain_id=self.brain_id",
            ],
            "src/core/agents/tools/kg_agent/KGAgentAddTripletsTool.py": [
                'store="nodes"',
                "brain_id=self.brain_id",
                "k=5",
            ],
        }
        for relative_path, snippets in expected_snippets.items():
            source = read_source(relative_path)
            for snippet in snippets:
                self.assertIn(snippet, source)
                
                
class JanitorToolDeduplicationTests(unittest.TestCase):
    def test_janitor_tools_init_exports_single_graph_read_tool(self):
        source = read_source("src/core/agents/tools/janitor_agent/__init__.py")
        self.assertIn("JanitorAgentExecuteGraphReadOperationTool", source)
        self.assertNotIn("JanitorAgentExecuteGraphOperationTool", source)

    def test_janitor_agent_imports_single_graph_read_tool(self):
        source = read_source("src/core/agents/janitor_agent.py")
        self.assertIn("JanitorAgentExecuteGraphReadOperationTool", source)
        self.assertNotIn("JanitorAgentExecuteGraphOperationTool", source)


if __name__ == "__main__":
    unittest.main()
