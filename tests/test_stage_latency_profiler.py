import asyncio
import contextlib
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.search import entity_info as entity_info_mod
from src.core.search.entity_info import EventSynergyRetriever
from src.lib.tracing.profiler import (
    STAGE_PROFILER_ENV_FLAG,
    profile_request,
    profile_stage,
    stage_profiling_enabled,
)
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, happened_at: str | None = None, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str, *, v_id: str | None = None):
    predicate = Predicate(uuid=uuid, name=name, description=f"desc-{name}")
    if v_id is not None:
        predicate.properties = {"v_id": v_id}
    return predicate


def _stages_by_name(report: dict) -> dict[str, dict]:
    return {stage["stage"]: stage for stage in report["stages"]}


def _facts(count: int):
    return [
        (
            _node("a", "Alice"),
            _pred(f"r{i}a", f"ATTENDED_{i}"),
            _node(
                f"e{i}",
                f"Event{i}",
                happened_at=f"2024-0{i + 1}-01",
                labels=["EVENT"],
            ),
            _pred(f"r{i}b", f"WITH_{i}"),
            _node(f"b{i}", f"Person{i}"),
        )
        for i in range(count)
    ]


@contextlib.contextmanager
def _stubbed_adapters(retrieve_mod, facts, extra=()):
    mock_embeddings = MagicMock()
    mock_embeddings.embed_text.return_value = Vector(
        id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
    )
    mock_vs = MagicMock()
    mock_vs.search_nodes.return_value = [
        Vector(
            id="n1",
            embeddings=[0.1, 0.2, 0.3],
            metadata={"uuid": "a", "name": "Alice"},
            distance=0.1,
        )
    ]
    mock_vs.search_relationships.return_value = []
    mock_vs.search_data.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_event_centric_neighbors.return_value = facts
    mock_data = MagicMock()
    mock_data.get_last_text_chunks.return_value = []
    mock_data.get_last_structured_data.return_value = []
    mock_data.search.return_value = MagicMock(text_chunks=[])
    mock_data.get_text_chunks_by_ids.return_value = ([], [])

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings)
        )
        stack.enter_context(patch.object(retrieve_mod, "vector_search", mock_vs))
        stack.enter_context(patch.object(retrieve_mod, "graph_adapter", mock_graph))
        stack.enter_context(patch.object(retrieve_mod, "data_adapter", mock_data))
        stack.enter_context(
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=ExtractElementsResponse(
                    tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
                    noun_chunks=[],
                ),
            )
        )
        for context in extra:
            stack.enter_context(context)
        yield mock_graph


class StageProfilerMechanismTests(unittest.TestCase):
    def test_spans_are_not_recorded_unless_a_profile_is_open(self):
        with profile_stage("orphan") as detail:
            detail["ignored"] = 1
        self.assertFalse(stage_profiling_enabled())
        with profile_request("noop", enabled=False) as profiler:
            self.assertIsNone(profiler)
            with profile_stage("still-off"):
                pass

    def test_environment_flag_enables_profiling_without_a_request_flag(self):
        with patch.dict(os.environ, {STAGE_PROFILER_ENV_FLAG: "true"}):
            self.assertTrue(stage_profiling_enabled())
            with profile_request("env", enabled=False, publish=False) as profiler:
                self.assertIsNotNone(profiler)
                with profile_stage("worked"):
                    pass
        self.assertIn("worked", _stages_by_name(profiler.last_report))

    def test_nested_spans_carry_their_parent_and_detail(self):
        with profile_request("root", enabled=True, publish=False) as profiler:
            with profile_stage("outer"):
                with profile_stage("inner", calls_expected=1) as detail:
                    detail["rows"] = 7

        stages = _stages_by_name(profiler.last_report)
        self.assertEqual(stages["outer"]["parent"], "root")
        self.assertEqual(stages["inner"]["parent"], "outer")
        self.assertEqual(stages["inner"]["detail"]["rows"], 7)

    def test_repeated_serial_spans_aggregate_without_overlap(self):
        with profile_request("root", enabled=True, publish=False) as profiler:
            for _ in range(3):
                with profile_stage("serial"):
                    time.sleep(0.01)

        stage = _stages_by_name(profiler.last_report)["serial"]
        self.assertEqual(stage["calls"], 3)
        self.assertEqual(stage["overlap_ms"], 0.0)
        self.assertAlmostEqual(stage["wall_ms"], stage["wall_sum_ms"], delta=1.0)

    def test_concurrent_spans_report_overlap_instead_of_a_false_sum(self):
        async def scenario():
            with profile_request("root", enabled=True, publish=False) as profiler:
                with profile_stage("fanout", blocking=False):
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(_sleep_span, "branch", 0.05)
                            for _ in range(4)
                        ]
                    )
            return profiler.last_report

        report = asyncio.run(scenario())
        stages = _stages_by_name(report)
        branch = stages["branch"]
        self.assertEqual(branch["calls"], 4)
        self.assertEqual(branch["parent"], "fanout")
        self.assertGreater(branch["threads"], 1)
        self.assertGreater(branch["overlap_ms"], 0.0)
        self.assertLess(branch["wall_ms"], branch["wall_sum_ms"])
        self.assertLessEqual(branch["wall_ms"], stages["fanout"]["wall_ms"] + 1.0)

    def test_off_loop_work_is_excluded_from_the_event_loop_budget(self):
        async def scenario():
            with profile_request("root", enabled=True, publish=False) as profiler:
                with profile_stage("on_loop"):
                    _spin(0.03)
                await asyncio.to_thread(_sleep_span, "off_loop", 0.05)
            return profiler.last_report

        report = asyncio.run(scenario())
        stages = _stages_by_name(report)
        self.assertTrue(stages["on_loop"]["on_loop"])
        self.assertFalse(stages["off_loop"]["on_loop"])
        self.assertGreaterEqual(report["loop_blocked_ms"], 25.0)
        self.assertLess(report["loop_blocked_ms"], stages["off_loop"]["wall_ms"] + 40.0)

    def test_io_wait_is_separated_from_python_cpu_time(self):
        with profile_request("root", enabled=True, publish=False) as profiler:
            with profile_stage("waiting"):
                time.sleep(0.05)
            with profile_stage("computing"):
                _spin(0.05)

        stages = _stages_by_name(profiler.last_report)
        self.assertGreater(stages["waiting"]["io_wait_ms"], 30.0)
        self.assertLess(stages["waiting"]["cpu_ms"], 20.0)
        self.assertGreater(stages["computing"]["cpu_ms"], 20.0)

    def test_awaiting_spans_do_not_claim_cpu_time(self):
        async def scenario():
            with profile_request("root", enabled=True, publish=False) as profiler:
                with profile_stage("awaits", blocking=False):
                    await asyncio.sleep(0.01)
            return profiler.last_report

        stage = _stages_by_name(asyncio.run(scenario()))["awaits"]
        self.assertIsNone(stage["cpu_ms"])
        self.assertIsNone(stage["io_wait_ms"])

    def test_a_raising_body_still_records_the_span(self):
        with self.assertRaises(RuntimeError):
            with profile_request("root", enabled=True, publish=False) as profiler:
                with profile_stage("boom"):
                    raise RuntimeError("boom")

        self.assertIn("boom", _stages_by_name(profiler.last_report))


class GetContextStageCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_timings_are_attached_unless_the_caller_asks(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(3)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a")
            )

        self.assertIsNone(response.stage_timings)

    async def test_every_documented_stage_is_measured(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            _facts(3),
            extra=[
                patch.object(
                    retrieve_mod,
                    "_build_adjacency_from_seeds",
                    MagicMock(return_value={"a": ["e0"], "e0": ["a"]}),
                ),
                patch.object(
                    retrieve_mod,
                    "_context_looks_insufficient",
                    MagicMock(return_value=True),
                ),
            ],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Which books did Alice read?",
                    brain_id="brain-a",
                    sufficiency_retry=True,
                    profile_stages=True,
                )
            )

        report = response.stage_timings
        self.assertIsNotNone(report)
        stages = _stages_by_name(report)
        for expected in (
            "nlp.extract_elements",
            "nlp.query_variants",
            "retrieval.fanout",
            "facts.variant",
            "facts.seed_search",
            "embed.query",
            "vector.search_nodes",
            "vector.search_relationships",
            "graph.event_neighbors",
            "passages.collect",
            "passages.retrieve",
            "vector.search_data",
            "data.keyword_search",
            "historical.context",
            "facts.dedup_rank",
            "ppr",
            "ppr.adjacency",
            "ppr.iterations",
            "fact_filter",
            "context.assemble",
            "sufficiency.check",
            "sufficiency.retry",
            "context.render",
        ):
            self.assertIn(expected, stages, f"{expected} was not measured")
        self.assertFalse(any(name.startswith("dossiers") for name in stages))
        self.assertGreater(report["total_ms"], 0.0)
        self.assertGreaterEqual(report["span_count"], len(stages))

    async def test_the_concurrent_fanout_is_reported_as_overlapping(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(2)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="What did Alice and Bob do together?",
                    brain_id="brain-a",
                    profile_stages=True,
                )
            )

        stages = _stages_by_name(response.stage_timings)
        fanout = stages["retrieval.fanout"]
        variants = stages["facts.variant"]
        self.assertGreater(variants["calls"], 1)
        self.assertLessEqual(variants["wall_ms"], fanout["wall_ms"] + 1.0)
        self.assertLessEqual(fanout["wall_ms"], response.stage_timings["total_ms"])

    async def test_ppr_stages_are_attributed_to_the_event_loop(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            _facts(3),
            extra=[
                patch.object(
                    retrieve_mod,
                    "_build_adjacency_from_seeds",
                    MagicMock(return_value={"a": ["e0"], "e0": ["a"]}),
                )
            ],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", profile_stages=True
                )
            )

        stages = _stages_by_name(response.stage_timings)
        self.assertTrue(stages["ppr.adjacency"]["on_loop"])
        self.assertTrue(stages["ppr.iterations"]["on_loop"])
        self.assertTrue(stages["nlp.extract_elements"]["on_loop"])
        self.assertIsNone(stages["nlp.extract_elements"]["cpu_ms"])
        self.assertFalse(stages["facts.variant"]["on_loop"])
        self.assertEqual(stages["ppr.iterations"]["detail"]["iterations"], 20)

    async def test_the_report_is_published_as_a_latency_trace_event(self):
        from src.lib.tracing.events import TraceEventType
        from src.lib.tracing.tracker import tracer
        from src.services.api.controllers import retrieve as retrieve_mod

        tracer.queue.drain()
        with _stubbed_adapters(retrieve_mod, _facts(1)):
            await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", profile_stages=True
                )
            )

        events = [
            event
            for event in tracer.queue.drain()
            if event.event_type is TraceEventType.LATENCY
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "retrieve.context.stages")
        self.assertIn("stages", events[0].metadata)


class SpacyStagePassesTests(unittest.TestCase):
    def test_the_duplicate_process_doc_pass_is_visible_in_the_report(self):
        from src.utils.nlp import ner as ner_mod

        with (
            patch.object(ner_mod, "langid_detect", return_value=("en", 1.0)),
            patch.object(
                ner_mod._entity_extractor.spacy_manager,
                "get_model",
                return_value=MagicMock(return_value=object()),
            ),
            patch.object(
                ner_mod.MultiLangEntityExtractor,
                "_process_doc",
                return_value={"tokens": [], "noun_chunks": []},
            ),
        ):
            with profile_request("nlp", enabled=True, publish=False) as profiler:
                ner_mod._entity_extractor.extract_elements("Alice read a book")

        stages = _stages_by_name(profiler.last_report)
        self.assertEqual(stages["nlp.parse"]["calls"], 1)
        self.assertEqual(stages["nlp.process_doc"]["calls"], 2)
        self.assertTrue(stages["nlp.process_doc"]["on_loop"])


class DossierWalkStageTests(unittest.TestCase):
    def test_one_vector_fetch_per_edge_is_counted(self):
        nodes = {name: _node(name, name.upper()) for name in ("a", "b", "c", "d")}
        rels = {
            "a": [
                (_pred("r1", "TO_B", v_id="v1"), nodes["b"]),
                (_pred("r2", "TO_C", v_id="v2"), nodes["c"]),
            ],
            "b": [(_pred("r3", "TO_D", v_id="v3"), nodes["d"])],
            "c": [],
            "d": [],
        }
        edges = sum(len(v) for v in rels.values())

        graph = MagicMock()
        graph.get_neighbors.side_effect = lambda ids, brain_id="default", **kwargs: {
            ids[0]: rels.get(ids[0], [])
        }
        vectors = MagicMock()
        vectors.get_by_ids.side_effect = lambda ids, **kwargs: [
            Vector(id=ids[0], embeddings=[1.0, 0.0], metadata={})
        ]

        with (
            patch.object(entity_info_mod, "graph_adapter", graph),
            patch.object(entity_info_mod, "vector_store_adapter", vectors),
            patch.object(entity_info_mod, "cosine_similarity", return_value=0.5),
        ):
            retriever = EventSynergyRetriever("brain-a")
            with profile_request("dossier", enabled=True, publish=False) as profiler:
                with profile_stage("dossiers.explore"):
                    retriever._recursive_explorer(
                        "a", [1.0, 0.0], depth=3, visited_ids=set(), work_counter=[0]
                    )

        stages = _stages_by_name(profiler.last_report)
        fetches = stages["dossiers.edge_vector_fetch"]
        self.assertEqual(fetches["calls"], edges)
        self.assertEqual(fetches["parent"], "dossiers.explore")
        self.assertEqual(stages["dossiers.get_neighbors"]["parent"], "dossiers.explore")


def _sleep_span(name: str, seconds: float) -> None:
    with profile_stage(name):
        time.sleep(seconds)


def _spin(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        threading.get_ident()


if __name__ == "__main__":
    unittest.main()
