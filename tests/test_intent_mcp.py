import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.data import TextChunk
from src.constants.kg import SearchEntitiesResult, SearchRelationshipsResult, Triple
from src.core.intent.parse import tokenize
from src.core.intent.brain import BrainResolutionError, resolve_brain
from src.core.intent.parse import parse_fact_text, parse_fact_text_regex
from src.core.intent.retire import RetireLookupError, apply_open_update
from src.core.intent.service import check_claim, correct_fact, recall_facts, remember_fact

_LLM_DISABLED = patch("src.core.intent.parse._configured_llm", return_value=None)
_LLM_DISABLED.start()
from src.core.verification.facts import UNVERIFIED
from src.core.verification.service import FactNotFound, apply_verification_action
from src.services.mcp.intent_tools import intent_tool_descriptions, intent_tool_token_count
from src.services.mcp.prompt import MCP_TOOL_SPECS, build_mcp_tools_instructions


class FakeBrain:
    def __init__(self, name_key, pat):
        self.name_key = name_key
        self.pat = pat


class FakeData:
    def __init__(self, brains=None):
        self.chunks = []
        self.brains = brains or []

    def save_text_chunk(self, text_chunk, brain_id="default"):
        self.chunks.append(text_chunk)
        return text_chunk

    def get_text_chunks(
        self,
        brain_id="default",
        limit=10,
        skip=0,
        query_text=None,
        metadata_eq=None,
        order="desc",
    ):
        items = list(self.chunks)
        if metadata_eq:
            items = [
                chunk
                for chunk in items
                if all((chunk.metadata or {}).get(key) == value for key, value in metadata_eq.items())
            ]
        if query_text:
            needle = query_text.lower()
            items = [chunk for chunk in items if needle in (chunk.text or "").lower()]
        return items[skip : skip + limit], len(items)

    def mark_text_chunks_superseded(
        self, ids, superseded_at, brain_id, superseded_by=None, valid_to=None
    ):
        wanted = {str(item) for item in ids}
        for chunk in self.chunks:
            if str(chunk.id) not in wanted:
                continue
            meta = dict(chunk.metadata or {})
            meta["superseded_at"] = superseded_at
            if valid_to:
                meta["valid_to"] = valid_to
            if superseded_by:
                meta["superseded_by"] = superseded_by
            chunk.metadata = meta

    def get_brains_list(self):
        return list(self.brains)

    def get_open_conflicts_for_facts(self, fact_ids, brain_id):
        return []


class FakeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._mu = threading.RLock()
        self.before_mutate = None

    def _run_hook(self):
        hook = self.before_mutate
        if callable(hook):
            hook()

    def add_nodes(self, nodes, brain_id="default", identification_params=None, metadata=None):
        with self._mu:
            for node in nodes:
                self.nodes[node.uuid] = node
            return list(nodes)

    def add_relationship(self, subject, predicate, to_object, brain_id="default"):
        self._run_hook()
        with self._mu:
            self.nodes[subject.uuid] = subject
            self.nodes[to_object.uuid] = to_object
            predicate.properties = dict(predicate.properties or {})
            predicate.properties.setdefault("uuid", predicate.uuid)
            self.edges.append((subject.uuid, predicate, to_object.uuid))
            return predicate.uuid

    def get_neighbors(
        self,
        nodes,
        brain_id="default",
        same_type_only=False,
        limit=None,
        of_types=None,
    ):
        self._run_hook()
        with self._mu:
            uuids = [node if isinstance(node, str) else node.uuid for node in nodes]
            neighbors = {uuid: [] for uuid in uuids}
            for subject_uuid, predicate, object_uuid in self.edges:
                if subject_uuid in neighbors:
                    neighbors[subject_uuid].append(
                        (
                            predicate.model_copy(update={"direction": "out"}),
                            self.nodes[object_uuid],
                        )
                    )
                if object_uuid in neighbors:
                    neighbors[object_uuid].append(
                        (
                            predicate.model_copy(update={"direction": "in"}),
                            self.nodes[subject_uuid],
                        )
                    )
            return neighbors

    def update_properties(
        self,
        uuid,
        updating,
        brain_id="default",
        new_properties=None,
        properties_to_remove=None,
        expected_open=False,
    ):
        with self._mu:
            for _subject, predicate, _obj in self.edges:
                if predicate.uuid == uuid:
                    current = predicate.properties or {}
                    if expected_open and (current.get("invalid_at") or current.get("valid_to")):
                        return None
                    predicate.properties = {
                        **current,
                        **(new_properties or {}),
                    }
                    if (new_properties or {}).get("deprecated"):
                        predicate.deprecated = True
                    return predicate
            return None

    def _row(self, subject_uuid, predicate, object_uuid):
        subject = self.nodes[subject_uuid]
        obj = self.nodes[object_uuid]
        props = dict(predicate.properties or {})
        props.setdefault("uuid", predicate.uuid)
        return {
            "kind": "relationship",
            "uuid": predicate.uuid,
            "name": predicate.name,
            "description": predicate.description,
            "properties": props,
            "subject_uuid": subject_uuid,
            "subject_name": subject.name,
            "subject_labels": subject.labels,
            "object_uuid": object_uuid,
            "object_name": obj.name,
            "object_labels": obj.labels,
            "labels": [],
        }

    def get_fact_record(self, fact_id, brain_id="default"):
        with self._mu:
            for subject_uuid, predicate, object_uuid in self.edges:
                if predicate.uuid == fact_id:
                    return self._row(subject_uuid, predicate, object_uuid)
            return None

    def set_fact_verification(
        self, fact_id, brain_id="default", *, kind, properties, history_entry=None
    ):
        from src.core.verification.facts import serialize_history_entry

        with self._mu:
            for subject_uuid, predicate, object_uuid in self.edges:
                if predicate.uuid != fact_id:
                    continue
                props = {**(predicate.properties or {}), **properties}
                entry = serialize_history_entry(history_entry)
                if entry:
                    history = list(props.get("verification_history") or [])
                    history.append(entry)
                    props["verification_history"] = history
                predicate.properties = props
                if properties.get("verification_state"):
                    predicate.verification_state = properties["verification_state"]
                return self._row(subject_uuid, predicate, object_uuid)
            return None

    def list_facts_by_verification(
        self,
        brain_id="default",
        state=None,
        include_rejected=False,
        limit=50,
        skip=0,
    ):
        rows = []
        for subject_uuid, predicate, object_uuid in self.edges:
            row = self._row(subject_uuid, predicate, object_uuid)
            current = (row["properties"].get("verification_state") or UNVERIFIED)
            if state and current != state:
                continue
            if not include_rejected and current == "rejected":
                continue
            rows.append(row)
        return rows[skip : skip + limit], len(rows)

    def search_entities(
        self,
        brain_id="default",
        limit=10,
        skip=0,
        node_labels=None,
        query_text=None,
    ):
        nodes = list(self.nodes.values())
        if node_labels:
            wanted = {str(label).upper() for label in node_labels}
            nodes = [
                node
                for node in nodes
                if wanted.intersection({str(label).upper() for label in (node.labels or [])})
            ]
        if query_text:
            terms = tokenize(query_text)
            nodes = [
                node
                for node in nodes
                if tokenize(node.name) & terms or query_text.lower() in (node.name or "").lower()
            ]
        return SearchEntitiesResult(results=nodes[skip : skip + limit], total=len(nodes))

    def search_relationships(
        self,
        brain_id="default",
        limit=10,
        skip=0,
        relationship_types=None,
        from_node_labels=None,
        to_node_labels=None,
        query_text=None,
        query_search_target="all",
    ):
        triples = []
        for subject_uuid, predicate, object_uuid in self.edges:
            if relationship_types and predicate.name not in relationship_types:
                continue
            subject = self.nodes[subject_uuid]
            obj = self.nodes[object_uuid]
            blob = " ".join(
                part
                for part in (
                    subject.name,
                    predicate.name,
                    obj.name,
                    predicate.description,
                    (predicate.properties or {}).get("fact_text"),
                )
                if part
            )
            if query_text:
                terms = tokenize(query_text)
                if not (terms & tokenize(blob)) and query_text.lower() not in blob.lower():
                    continue
            triples.append(Triple(subject=subject, predicate=predicate, object=obj))
        return SearchRelationshipsResult(
            results=triples[skip : skip + limit], total=len(triples)
        )


def _guard(token, target=None):
    if token == "pat-alpha":
        if target and target != "alpha":
            return False
        return "alpha"
    if token == "system":
        return True
    return False


class IntentParseTests(unittest.TestCase):
    def test_lives_in_and_is(self):
        lives = parse_fact_text("Alice lives in Paris", llm=None)
        self.assertEqual(lives.subject, "Alice")
        self.assertEqual(lives.predicate, "LIVES_IN")
        self.assertEqual(lives.object, "Paris")
        status = parse_fact_text("Refund window is 30 days", llm=None)
        self.assertEqual(status.predicate, "IS")
        self.assertEqual(status.object, "30 days")
        regex = parse_fact_text_regex("Alice lives in Paris")
        self.assertEqual(regex.predicate, "LIVES_IN")


class IntentLlmParseTests(unittest.TestCase):
    def test_non_english_uses_llm(self):
        llm = MagicMock()
        llm.generate_json.return_value = {
            "subject": "Alice",
            "predicate": "LIVES_IN",
            "object": "Roma",
            "subject_type": "PERSON",
            "object_type": "PLACE",
        }
        parsed = parse_fact_text("Alice vive a Roma", llm=llm)
        self.assertEqual(parsed.subject, "Alice")
        self.assertEqual(parsed.predicate, "LIVES_IN")
        self.assertEqual(parsed.object, "Roma")
        self.assertEqual(parsed.subject_type, "PERSON")
        self.assertEqual(parsed.object_type, "PLACE")
        llm.generate_json.assert_called_once()

    def test_fallback_when_llm_fails(self):
        llm = MagicMock()
        llm.generate_json.side_effect = RuntimeError("extractor unavailable")
        parsed = parse_fact_text("Alice lives in Paris", llm=llm)
        self.assertEqual(parsed.subject, "Alice")
        self.assertEqual(parsed.predicate, "LIVES_IN")
        self.assertEqual(parsed.object, "Paris")

    def test_fallback_when_llm_returns_incomplete(self):
        llm = MagicMock()
        llm.generate_json.return_value = {"subject": "Alice"}
        parsed = parse_fact_text("Alice works at Acme", llm=llm)
        self.assertEqual(parsed.predicate, "WORKS_AT")
        self.assertEqual(parsed.object, "Acme")

    def test_remember_uses_llm_extraction(self):
        graph = FakeGraph()
        data = FakeData([FakeBrain("alpha", "pat-alpha")])
        llm = MagicMock()
        llm.generate_json.return_value = {
            "subject": "Alice",
            "predicate": "LIVES_IN",
            "object": "Roma",
            "subject_type": "PERSON",
            "object_type": "PLACE",
        }
        with patch("src.core.intent.parse._configured_llm", return_value=llm):
            result = remember_fact(
                fact="Alice vive a Roma",
                source="handbook.md",
                brain_id="alpha",
                token="pat-alpha",
                graph=graph,
                data=data,
            )
        self.assertEqual(result.fact.subject, "Alice")
        self.assertEqual(result.fact.predicate, "LIVES_IN")
        self.assertEqual(result.fact.object, "Roma")


class BrainResolutionTests(unittest.TestCase):
    def test_token_default_brain_and_errors(self):
        data = FakeData([FakeBrain("alpha", "pat-alpha"), FakeBrain("beta", "pat-beta")])
        self.assertEqual(resolve_brain("pat-alpha", None, data_adapter=data, guard=_guard), "alpha")
        with self.assertRaises(BrainResolutionError) as raised:
            resolve_brain("system", None, data_adapter=data, guard=_guard)
        self.assertEqual(raised.exception.code, "ambiguous_brain")
        with self.assertRaises(BrainResolutionError) as missing:
            resolve_brain(None, None, data_adapter=data, guard=_guard)
        self.assertEqual(missing.exception.code, "unauthorized")
        with self.assertRaises(BrainResolutionError) as empty:
            resolve_brain("system", None, data_adapter=FakeData([]), guard=_guard)
        self.assertEqual(empty.exception.code, "no_brain")


class IntentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph()
        self.data = FakeData([FakeBrain("alpha", "pat-alpha")])
        self.token = "pat-alpha"
        self.brain_id = "alpha"

    def _remember(self, fact, source="handbook.md", **kwargs):
        return remember_fact(
            fact=fact,
            source=source,
            brain_id=self.brain_id,
            token=self.token,
            graph=self.graph,
            data=self.data,
            **kwargs,
        )

    def test_remember_lands_unverified_with_writer_and_source(self):
        result = self._remember("Alice lives in Paris", source="https://example.com/hr")
        self.assertEqual(result.fact.status, "unverified")
        self.assertEqual(result.fact.verification_state, UNVERIFIED)
        self.assertTrue(result.fact.writer.startswith("alpha:"))
        self.assertEqual(result.fact.source.url, "https://example.com/hr")
        self.assertIn("unverified", result.text)

    def test_recall_as_of_before_and_after_correction(self):
        self._remember(
            "Alice lives in Paris",
            valid_from="2024-01-01T00:00:00Z",
            source_key="alice-home",
        )
        correct_fact(
            reason="moved",
            brain_id=self.brain_id,
            token=self.token,
            text="Alice lives in Paris",
            correction="Alice lives in London",
            graph=self.graph,
            data=self.data,
        )
        before = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            as_of="2024-06-01T00:00:00Z",
            graph=self.graph,
            data=self.data,
        )
        after = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertTrue(before.facts)
        self.assertEqual(before.facts[0].object, "Paris")
        self.assertEqual(before.facts[0].status, "superseded")
        self.assertEqual(before.facts[0].retired_reason, "moved")
        self.assertTrue(after.facts)
        self.assertEqual(after.facts[0].object, "London")
        self.assertEqual(after.facts[0].status, "unverified")

    def test_check_rejected_claim_is_contradicted(self):
        remembered = self._remember("Refund window is 30 days")
        apply_verification_action(
            self.graph,
            fact_id=remembered.fact.fact_id,
            brain_id=self.brain_id,
            action="reject",
            verified_by="analyst",
            verification_source="policy-desk",
            note="replaced by 14-day window",
        )
        checked = check_claim(
            claim="Refund window is 30 days",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(checked.verdict, "contradicted")
        self.assertEqual(checked.evidence[0].retired_reason, "replaced by 14-day window")
        self.assertEqual(checked.evidence[0].source.doc, "handbook.md")

    def test_verified_facts_keep_rank_boost(self):
        first = self._remember("Alice lives in Paris")
        second = self._remember("Alice works at Acme")
        apply_verification_action(
            self.graph,
            fact_id=second.fact.fact_id,
            brain_id=self.brain_id,
            action="verify",
            verified_by="analyst",
            verification_source="hr",
        )
        recalled = recall_facts(
            question="Alice",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(recalled.facts[0].fact_id, second.fact.fact_id)
        self.assertEqual(recalled.facts[0].status, "verified")
        self.assertEqual(first.fact.status, "unverified")

    def test_graphiti_1728_unrelated_facts_are_not_retired(self):
        home = self._remember("Alice lives in Paris", source_key="note-1")
        job = self._remember("Alice works at Acme", source_key="note-1")
        other = self._remember("Bob lives in Lyon", source_key="note-1")
        correct_fact(
            reason="moved cities",
            brain_id=self.brain_id,
            token=self.token,
            fact_id=home.fact.fact_id,
            correction="Alice lives in London",
            graph=self.graph,
            data=self.data,
        )
        job_row = self.graph.get_fact_record(job.fact.fact_id, self.brain_id)
        other_row = self.graph.get_fact_record(other.fact.fact_id, self.brain_id)
        home_row = self.graph.get_fact_record(home.fact.fact_id, self.brain_id)
        self.assertFalse((job_row["properties"] or {}).get("invalid_at"))
        self.assertFalse((other_row["properties"] or {}).get("invalid_at"))
        self.assertTrue((home_row["properties"] or {}).get("invalid_at"))
        self.assertEqual((home_row["properties"] or {}).get("retired_reason"), "moved cities")

    def test_correct_without_correction_keeps_history(self):
        remembered = self._remember("Alice lives in Paris")
        result = correct_fact(
            reason="no longer true",
            brain_id=self.brain_id,
            token=self.token,
            fact_id=remembered.fact.fact_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(result.fact.status, "rejected")
        self.assertEqual(result.previous.fact_id, remembered.fact.fact_id)
        history = self.graph.get_fact_record(remembered.fact.fact_id, self.brain_id)["properties"]["verification_history"]
        self.assertTrue(history)

    def _validity(self, fact_id):
        row = self.graph.get_fact_record(fact_id, self.brain_id)
        props = row["properties"] or {}
        return props.get("valid_from") or props.get("valid_at"), props.get("valid_to") or props.get("invalid_at")

    def test_remember_newer_then_older_keeps_newer_current(self):
        newer = self._remember(
            "Alice lives in Milan",
            valid_from="2024-01-01T00:00:00Z",
        )
        older = self._remember(
            "Alice lives in Rome",
            valid_from="2020-01-01T00:00:00Z",
        )
        older_from, older_to = self._validity(older.fact.fact_id)
        newer_from, newer_to = self._validity(newer.fact.fact_id)
        self.assertEqual(older_from, "2020-01-01T00:00:00Z")
        self.assertEqual(older_to, "2024-01-01T00:00:00Z")
        self.assertGreaterEqual(older_to, older_from)
        self.assertFalse(newer_to)
        self.assertEqual(newer_from, "2024-01-01T00:00:00Z")
        recalled = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(recalled.facts[0].object, "Milan")
        historical = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            as_of="2021-06-01T00:00:00Z",
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(historical.facts[0].object, "Rome")

    def test_remember_older_then_newer_retires_older(self):
        older = self._remember(
            "Alice lives in Rome",
            valid_from="2020-01-01T00:00:00Z",
        )
        newer = self._remember(
            "Alice lives in Milan",
            valid_from="2024-01-01T00:00:00Z",
        )
        older_from, older_to = self._validity(older.fact.fact_id)
        newer_from, newer_to = self._validity(newer.fact.fact_id)
        self.assertEqual(older_to, "2024-01-01T00:00:00Z")
        self.assertGreaterEqual(older_to, older_from)
        self.assertFalse(newer_to)
        self.assertEqual(newer_from, "2024-01-01T00:00:00Z")

    def test_remember_equal_dates_replaces_without_inverted_window(self):
        first = self._remember(
            "Alice lives in Rome",
            valid_from="2024-01-01T00:00:00Z",
        )
        second = self._remember(
            "Alice lives in Milan",
            valid_from="2024-01-01T00:00:00Z",
        )
        first_from, first_to = self._validity(first.fact.fact_id)
        second_from, second_to = self._validity(second.fact.fact_id)
        self.assertEqual(first_from, "2024-01-01T00:00:00Z")
        self.assertGreaterEqual(first_to, first_from)
        self.assertFalse(second_to)
        self.assertEqual(second_from, "2024-01-01T00:00:00Z")
        live = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(live.facts[0].object, "Milan")

    def test_source_key_rewrite_supersedes_same_scope_only(self):
        first = self._remember(
            "Alice lives in Paris",
            source_key="alice-home",
            valid_from="2024-01-01T00:00:00Z",
        )
        second = self._remember(
            "Alice lives in London",
            source_key="alice-home",
            valid_from="2025-01-01T00:00:00Z",
        )
        old = self.graph.get_fact_record(first.fact.fact_id, self.brain_id)
        self.assertEqual((old["properties"] or {}).get("superseded_by"), second.fact.fact_id)
        self.assertEqual((old["properties"] or {}).get("retired_reason"), "source_key rewrite")

    def test_check_does_not_support_another_subject(self):
        self._remember("Bob lives in Paris")
        checked = check_claim(
            claim="Alice lives in Paris",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(checked.verdict, "unknown")

    def test_correct_text_requires_unique_match(self):
        self._remember("Bob lives in Paris")
        with self.assertRaises((FactNotFound, ValueError)):
            correct_fact(
                reason="no longer true",
                brain_id=self.brain_id,
                token=self.token,
                text="Alice lives in Paris",
                graph=self.graph,
                data=self.data,
            )

    def test_correct_keeps_employed_by_predicate(self):
        remembered = self._remember("Alice works for Acme")
        result = correct_fact(
            reason="new employer",
            brain_id=self.brain_id,
            token=self.token,
            fact_id=remembered.fact.fact_id,
            correction="Alice works for Contoso",
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(result.fact.predicate, "EMPLOYED_BY")
        self.assertEqual(result.fact.object, "Contoso")

    def test_correct_does_not_retire_sibling_non_functional_facts(self):
        python = self._remember("Build uses Python")
        docker = self._remember("Build uses Docker")
        correct_fact(
            reason="pin version",
            brain_id=self.brain_id,
            token=self.token,
            fact_id=python.fact.fact_id,
            correction="Build uses Python 3",
            graph=self.graph,
            data=self.data,
        )
        docker_row = self.graph.get_fact_record(docker.fact.fact_id, self.brain_id)
        self.assertFalse((docker_row["properties"] or {}).get("invalid_at"))

    def test_unchanged_source_key_reuses_passage(self):
        self._remember("Alice lives in Paris", source_key="alice-home")
        self._remember("Alice lives in Paris", source_key="alice-home")
        self.assertEqual(len(self.data.chunks), 1)

    def test_recall_hides_passage_for_rejected_graph_fact(self):
        remembered = self._remember("Alice lives in Paris")
        apply_verification_action(
            self.graph,
            fact_id=remembered.fact.fact_id,
            brain_id=self.brain_id,
            action="reject",
            verified_by="analyst",
            verification_source="handbook.md",
            note="no longer true",
        )
        recalled = recall_facts(
            question="Alice lives in Paris",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertFalse(recalled.facts)

    def test_check_as_of_supports_fact_later_superseded(self):
        self._remember(
            "Alice lives in Paris",
            valid_from="2024-01-01T00:00:00Z",
            source_key="alice-home",
        )
        correct_fact(
            reason="moved",
            brain_id=self.brain_id,
            token=self.token,
            text="Alice lives in Paris",
            correction="Alice lives in London",
            graph=self.graph,
            data=self.data,
        )
        checked = check_claim(
            claim="Alice lives in Paris",
            as_of="2024-06-01T00:00:00Z",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(checked.verdict, "supported")
        self.assertEqual(checked.evidence[0].object, "Paris")

    def test_passage_without_valid_from_uses_inserted_at(self):
        self.data.chunks.append(
            TextChunk(
                text="legacy note",
                metadata={"subject": "Legacy", "predicate": "STATES", "object": "recorded"},
                inserted_at=datetime(2025, 6, 1),
            )
        )
        recalled = recall_facts(
            question="legacy note",
            brain_id=self.brain_id,
            as_of="2024-06-01T00:00:00Z",
            graph=self.graph,
            data=self.data,
        )
        self.assertFalse(recalled.facts)

    def test_recall_unrelated_query_is_empty(self):
        self._remember("Alice lives in Paris")
        recalled = recall_facts(
            question="quantum chromodynamics lattice spacing",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(recalled.facts, [])
        self.assertFalse(recalled.matched)
        self.assertIn("No matching facts", recalled.text)

    def test_recall_does_not_depend_on_verification_scan_cap(self):
        remembered = self._remember("Alice lives in Paris")

        def empty_scan(*_args, **_kwargs):
            return [], 0

        self.graph.list_facts_by_verification = empty_scan
        recalled = recall_facts(
            question="where does Alice live",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertTrue(recalled.matched)
        self.assertEqual(recalled.facts[0].fact_id, remembered.fact.fact_id)
        self.assertEqual(recalled.facts[0].object, "Paris")

    def test_check_unrelated_claim_is_unknown_without_evidence(self):
        self._remember("Alice lives in Paris")
        checked = check_claim(
            claim="Refund window is 30 days",
            brain_id=self.brain_id,
            graph=self.graph,
            data=self.data,
        )
        self.assertEqual(checked.verdict, "unknown")
        self.assertFalse(checked.matched)
        self.assertEqual(checked.evidence, [])

    def test_recall_uses_vector_seed_path_when_available(self):
        remembered = self._remember("Alice lives in Paris")
        seed_uuid = None
        for row in self.graph.list_facts_by_verification(self.brain_id, include_rejected=True, limit=10, skip=0)[0]:
            if row["uuid"] == remembered.fact.fact_id:
                seed_uuid = row["subject_uuid"]
                break
        self.assertIsNotNone(seed_uuid)

        def fake_seeds(text, brain_id, embeddings=None):
            return [(seed_uuid, 0.01, "Alice")]

        with patch(
            "src.services.api.controllers.retrieve._seed_nodes_for_text",
            side_effect=fake_seeds,
        ):
            recalled = recall_facts(
                question="完全无关的问题",
                brain_id=self.brain_id,
                graph=self.graph,
                data=self.data,
            )
        self.assertTrue(recalled.matched)
        self.assertEqual(recalled.facts[0].object, "Paris")

    def test_retire_lookup_failure_is_not_empty_success(self):
        self._remember("Alice lives in Paris")

        def boom(*_args, **_kwargs):
            raise RuntimeError("graph down")

        self.graph.get_neighbors = boom
        with self.assertRaises(RetireLookupError):
            remember_fact(
                fact="Alice lives in London",
                source="handbook.md",
                brain_id=self.brain_id,
                token=self.token,
                graph=self.graph,
                data=self.data,
            )

    def test_apply_open_update_skips_already_closed(self):
        first = self._remember(
            "Alice lives in Paris",
            valid_from="2024-01-01T00:00:00Z",
        )
        apply_open_update(
            self.graph,
            first.fact.fact_id,
            self.brain_id,
            {"invalid_at": "2025-01-01T00:00:00Z", "valid_to": "2025-01-01T00:00:00Z"},
        )
        skipped = apply_open_update(
            self.graph,
            first.fact.fact_id,
            self.brain_id,
            {"invalid_at": "2020-01-01T00:00:00Z", "valid_to": "2020-01-01T00:00:00Z"},
        )
        self.assertFalse(skipped)
        _from, until = self._validity(first.fact.fact_id)
        self.assertEqual(until, "2025-01-01T00:00:00Z")
        self.assertGreaterEqual(until, _from)

    def test_concurrent_remember_leaves_one_current(self):
        self.graph.before_mutate = lambda: time.sleep(0.02)
        cities = ["Rome", "Milan", "Paris", "London"]

        def write(city):
            return remember_fact(
                fact=f"Alice lives in {city}",
                source="handbook.md",
                brain_id=self.brain_id,
                token=self.token,
                graph=self.graph,
                data=self.data,
                valid_from="2024-06-01T00:00:00Z",
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(write, city) for city in cities]
            wait(futures)
            for future in futures:
                future.result()
        current = []
        for _subject, predicate, object_uuid in self.graph.edges:
            if predicate.name != "LIVES_IN":
                continue
            props = predicate.properties or {}
            start = props.get("valid_from") or props.get("valid_at")
            end = props.get("valid_to") or props.get("invalid_at")
            if start and end:
                self.assertGreaterEqual(end, start)
            if not end:
                current.append(self.graph.nodes[object_uuid].name)
        self.assertEqual(len(current), 1)


class IntentToolSurfaceTests(unittest.TestCase):
    def test_intent_tools_are_listed_first_and_under_token_budget(self):
        names = [spec["name"] for spec in MCP_TOOL_SPECS]
        self.assertEqual(
            names[:4],
            ["brainapi_recall", "brainapi_check", "brainapi_remember", "brainapi_correct"],
        )
        self.assertIn("search_memory", names)
        self.assertIn("list_brains", names)
        self.assertLess(names.index("brainapi_recall"), names.index("search_memory"))
        self.assertLess(intent_tool_token_count(), 2777)
        combined = " ".join(description for _name, description in intent_tool_descriptions())
        self.assertIn("remember", combined.lower())
        self.assertIn("as of", combined.lower())
        self.assertIn("what changed", combined.lower())
        self.assertIn("who said", combined.lower())
        prompt = build_mcp_tools_instructions("alpha")
        self.assertIn("brainapi_recall", prompt)
        self.assertIn("brainapi_check", prompt)

    def test_plugin_register_is_idempotent(self):
        from src.services.mcp.intent_tools import register_intent_tools

        registered = []

        def capture(fn, **kwargs):
            registered.append((kwargs.get("name"), fn.__name__))

        import src.services.mcp.intent_tools as tools

        with patch.object(tools, "_REGISTERED", False):
            register_intent_tools(None, register=capture)
            register_intent_tools(None, register=capture)
        self.assertEqual(
            [name for name, _fn in registered],
            ["brainapi_recall", "brainapi_check", "brainapi_remember", "brainapi_correct"],
        )


class IntentMcpSchemaTests(unittest.TestCase):
    def test_fastmcp_exposes_output_schema_and_read_only_hints(self):
        from mcp.server import FastMCP

        import src.services.mcp.intent_tools as tools

        with patch.object(tools, "_REGISTERED", False):
            server = FastMCP("intent-schema")
            tools.register_intent_tools(server)
        listed = {item.name: item for item in server._tool_manager.list_tools()}
        self.assertEqual(
            list(listed)[:4],
            ["brainapi_recall", "brainapi_check", "brainapi_remember", "brainapi_correct"],
        )
        recall = listed["brainapi_recall"]
        self.assertIsNotNone(recall.output_schema)
        self.assertIn("facts", str(recall.output_schema))
        self.assertTrue(recall.annotations.readOnlyHint)
        self.assertTrue(listed["brainapi_check"].annotations.readOnlyHint)
        self.assertFalse(listed["brainapi_remember"].annotations.readOnlyHint)
        self.assertFalse(listed["brainapi_correct"].annotations.destructiveHint)
        self.assertTrue(listed["brainapi_correct"].annotations.idempotentHint)


class IntentMcpToolE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_tools_brainpat_without_brain_id_or_query_language(self):
        from src.services.mcp import intent_tools as tools

        graph = FakeGraph()
        data = FakeData([FakeBrain("alpha", "pat-alpha")])

        with patch.object(tools, "_graph_and_data", lambda: (graph, data)), patch.object(
            tools, "_auth_token", return_value="pat-alpha"
        ), patch.object(tools, "guard_brainpat", _guard):
            remembered = await tools.brainapi_remember(
                fact="Alice lives in Paris",
                source="hr-handbook",
                valid_from="2024-01-01T00:00:00Z",
            )
            await tools.brainapi_correct(
                reason="moved",
                text="Alice lives in Paris",
                correction="Alice lives in London",
            )
            as_of_answer = await tools.brainapi_recall(
                question="what is Alice's city as of 2024-06-01",
                as_of="2024-06-01T00:00:00Z",
            )
            still_true = await tools.brainapi_check(claim="Alice lives in Paris")

        remembered_payload = remembered.structuredContent
        as_of_payload = as_of_answer.structuredContent
        check_payload = still_true.structuredContent
        self.assertEqual(remembered_payload["fact"]["status"], "unverified")
        self.assertEqual(as_of_payload["facts"][0]["object"], "Paris")
        self.assertEqual(as_of_payload["facts"][0]["status"], "superseded")
        self.assertIn(check_payload["verdict"], {"contradicted", "superseded"})
        self.assertNotIn("MATCH", as_of_answer.content[0].text)
        self.assertNotIn("SELECT", as_of_answer.content[0].text)
        self.assertNotIn("brain_id", as_of_answer.content[0].text.lower())


class IntentAgentE2ETests(unittest.TestCase):
    def test_oauth_token_answers_as_of_and_still_true_without_brain_id_or_query_language(self):
        graph = FakeGraph()
        data = FakeData([FakeBrain("alpha", "pat-alpha")])
        token = "pat-alpha"
        brain_id = resolve_brain(token, None, data_adapter=data, guard=_guard)
        remember_fact(
            fact="Alice lives in Paris",
            source="hr-handbook",
            brain_id=brain_id,
            token=token,
            valid_from="2024-01-01T00:00:00Z",
            graph=graph,
            data=data,
        )
        correct_fact(
            reason="moved",
            brain_id=brain_id,
            token=token,
            text="Alice lives in Paris",
            correction="Alice lives in London",
            graph=graph,
            data=data,
        )
        as_of_answer = recall_facts(
            question="what is Alice's city as of 2024-06-01",
            brain_id=brain_id,
            as_of="2024-06-01T00:00:00Z",
            graph=graph,
            data=data,
        )
        still_true = check_claim(
            claim="Alice lives in Paris",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        self.assertEqual(as_of_answer.facts[0].object, "Paris")
        self.assertEqual(as_of_answer.facts[0].status, "superseded")
        self.assertIn(still_true.verdict, {"contradicted", "superseded"})
        self.assertNotIn("MATCH", as_of_answer.text)
        self.assertNotIn("SELECT", as_of_answer.text)


if __name__ == "__main__":
    unittest.main()
