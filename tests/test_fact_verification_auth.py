import os
import sys
import unittest
from types import SimpleNamespace

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
    "POSTGRES_PASSWORD": "test-password",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.data import Brain, Observation, TextChunk
from src.constants.kg import Node, Predicate
from src.core.auth.scopes import (
    DEFAULT_SCOPE,
    VERIFICATION_SCOPE,
    effective_pat_scopes,
    grantable_oauth_scopes,
    has_verification_scope,
)
from src.core.verification.facts import (
    REJECTED,
    UNVERIFIED,
    VERIFIED,
    chunk_ids_with_only_rejected_support,
    extract_verification,
    filter_items_with_rejected_support,
    filter_raw_operation_result,
    serialize_history_entry,
    without_reserved_verification_keys,
)
from src.lib.postgresql.networkx_client import NetworkXGraphClient
from src.core.verification.service import apply_verification_action
from src.lib.neo4j.client import Neo4jClient
from src.services.mcp.errors import McpToolError
from src.services.mcp.main import (
    _mutate_fact_sync,
    _resolve_conflict_sync,
    _search_memory_sync,
    auth_principal_var,
    auth_scopes_var,
    auth_token_var,
)


class ScopeModelTests(unittest.TestCase):
    def test_existing_brain_pat_does_not_include_verification(self):
        brain = Brain(name_key="demo")
        self.assertEqual(brain.pat_scopes, [])
        scopes = effective_pat_scopes(brain)
        self.assertEqual(scopes, [DEFAULT_SCOPE])
        self.assertFalse(has_verification_scope(scopes))

    def test_opt_in_pat_scopes_grant_verification(self):
        brain = Brain(name_key="demo", pat_scopes=[VERIFICATION_SCOPE])
        scopes = effective_pat_scopes(brain)
        self.assertIn(DEFAULT_SCOPE, scopes)
        self.assertTrue(has_verification_scope(scopes))

    def test_oauth_cannot_escalate_plain_pat_to_verification(self):
        granted = grantable_oauth_scopes(
            [DEFAULT_SCOPE, VERIFICATION_SCOPE],
            [],
            valid_scopes=[DEFAULT_SCOPE, VERIFICATION_SCOPE],
        )
        self.assertEqual(granted, [DEFAULT_SCOPE])
        self.assertFalse(has_verification_scope(granted))

    def test_oauth_grants_verification_when_pat_holds_it(self):
        granted = grantable_oauth_scopes(
            [DEFAULT_SCOPE, VERIFICATION_SCOPE],
            [VERIFICATION_SCOPE],
            valid_scopes=[DEFAULT_SCOPE, VERIFICATION_SCOPE],
        )
        self.assertTrue(has_verification_scope(granted))


class RawSearchMemoryFilterTests(unittest.TestCase):
    def test_drops_rejected_graph_rows_by_default(self):
        result = {
            "records": [
                {
                    "r": {
                        "uuid": "rel-bad",
                        "verification_state": REJECTED,
                    }
                },
                {
                    "r": {
                        "uuid": "rel-ok",
                        "verification_state": UNVERIFIED,
                    }
                },
            ],
            "keys": ["r"],
            "truncated": False,
        }
        filtered = filter_raw_operation_result(result, include_rejected=False)
        self.assertEqual(len(filtered["records"]), 1)
        self.assertEqual(filtered["records"][0]["r"]["uuid"], "rel-ok")

    def test_include_rejected_keeps_rows(self):
        result = {
            "records": [
                {"r": {"uuid": "rel-bad", "verification_state": REJECTED}},
            ]
        }
        filtered = filter_raw_operation_result(result, include_rejected=True)
        self.assertEqual(len(filtered["records"]), 1)


class ObservationSupportFilterTests(unittest.TestCase):
    def test_hides_chunks_whose_only_support_is_rejected(self):
        rejected_rel = Predicate(
            uuid="r-bad",
            name="KNOWS",
            description="knows",
            properties={
                "verification_state": REJECTED,
                "source_chunk_ids": ["chunk-1"],
            },
        )
        live_rel = Predicate(
            uuid="r-ok",
            name="KNOWS",
            description="knows",
            properties={
                "verification_state": VERIFIED,
                "source_chunk_ids": ["chunk-2"],
            },
        )
        triples = [
            (
                Node(uuid="a", name="A", labels=["PERSON"]),
                rejected_rel,
                Node(uuid="b", name="B", labels=["PERSON"]),
            ),
            (
                Node(uuid="a", name="A", labels=["PERSON"]),
                live_rel,
                Node(uuid="c", name="C", labels=["PERSON"]),
            ),
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([], triples)
        self.assertEqual(rejected_ids, {"chunk-1"})
        chunks = [
            TextChunk(id="chunk-1", text="rejected support"),
            TextChunk(id="chunk-2", text="live support"),
            TextChunk(id="chunk-3", text="no graph support"),
        ]
        observations = [
            Observation(text="from rejected", resource_id="chunk-1"),
            Observation(text="from live", resource_id="chunk-2"),
        ]
        kept_chunks = filter_items_with_rejected_support(chunks, rejected_ids)
        kept_obs = filter_items_with_rejected_support(observations, rejected_ids)
        self.assertEqual([item.id for item in kept_chunks], ["chunk-2", "chunk-3"])
        self.assertEqual([item.resource_id for item in kept_obs], ["chunk-2"])

    def test_person_endpoint_citation_does_not_inherit_relationship_rejection(self):
        person = Node(
            uuid="p1",
            name="Alice",
            labels=["PERSON"],
            properties={"source_chunk_ids": ["chunk-a"]},
        )
        rejected_rel = Predicate(
            uuid="r-bad",
            name="KNOWS",
            description="knows",
            properties={
                "verification_state": REJECTED,
                "source_chunk_ids": ["chunk-b"],
            },
        )
        triples = [
            (
                person,
                rejected_rel,
                Node(uuid="p2", name="Bob", labels=["PERSON"]),
            )
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([person], triples)
        self.assertEqual(rejected_ids, {"chunk-b"})

    def test_person_node_does_not_count_as_non_rejected_support(self):
        person = Node(
            uuid="p1",
            name="Alice",
            labels=["PERSON"],
            properties={
                "verification_state": UNVERIFIED,
                "source_chunk_ids": ["chunk-1"],
            },
        )
        rejected_rel = Predicate(
            uuid="r-bad",
            name="KNOWS",
            description="knows",
            properties={
                "verification_state": REJECTED,
                "source_chunk_ids": ["chunk-1"],
            },
        )
        triples = [
            (
                person,
                rejected_rel,
                Node(uuid="p2", name="Bob", labels=["PERSON"]),
            )
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([person], triples)
        self.assertEqual(rejected_ids, {"chunk-1"})

    def test_unverified_event_keeps_chunk_despite_rejected_relationship(self):
        event = Node(
            uuid="e1",
            name="Signed",
            labels=["EVENT"],
            properties={
                "verification_state": UNVERIFIED,
                "source_chunk_ids": ["chunk-1"],
            },
        )
        rejected_rel = Predicate(
            uuid="r-bad",
            name="MENTIONS",
            description="mentions",
            properties={
                "verification_state": REJECTED,
                "source_chunk_ids": ["chunk-1"],
            },
        )
        triples = [
            (
                event,
                rejected_rel,
                Node(uuid="p1", name="Alice", labels=["PERSON"]),
            )
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([event], triples)
        self.assertEqual(rejected_ids, set())

    def test_nested_source_chunk_ids_are_counted(self):
        rejected_rel = Predicate(
            uuid="r-nested",
            name="KNOWS",
            description="knows",
            properties={
                "properties": {
                    "verification_state": REJECTED,
                    "source_chunk_ids": ["chunk-nested"],
                }
            },
        )
        triples = [
            (
                Node(uuid="a", name="A", labels=["PERSON"]),
                rejected_rel,
                Node(uuid="b", name="B", labels=["PERSON"]),
            )
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([], triples)
        self.assertEqual(rejected_ids, {"chunk-nested"})

    def test_outer_and_nested_source_chunk_ids_are_both_counted(self):
        rejected_rel = Predicate(
            uuid="r-both",
            name="KNOWS",
            description="knows",
            properties={
                "verification_state": REJECTED,
                "source_chunk_ids": ["chunk-outer"],
                "properties": {
                    "source_chunk_ids": ["chunk-inner"],
                },
            },
        )
        triples = [
            (
                Node(uuid="a", name="A", labels=["PERSON"]),
                rejected_rel,
                Node(uuid="b", name="B", labels=["PERSON"]),
            )
        ]
        rejected_ids = chunk_ids_with_only_rejected_support([], triples)
        self.assertEqual(rejected_ids, {"chunk-outer", "chunk-inner"})

    def test_outer_verification_state_wins_over_nested(self):
        record = extract_verification(
            {
                "verification_state": REJECTED,
                "properties": {"verification_state": VERIFIED},
            }
        )
        self.assertEqual(record.state, REJECTED)

    def test_nested_verification_fills_when_outer_missing(self):
        record = extract_verification(
            {
                "properties": {"verification_state": REJECTED},
            }
        )
        self.assertEqual(record.state, REJECTED)

    def test_without_reserved_keys_strips_nested_properties(self):
        cleaned = without_reserved_verification_keys(
            {
                "verification_state": VERIFIED,
                "name": "Alice",
                "properties": {
                    "verification_state": VERIFIED,
                    "source_chunk_ids": ["chunk-1"],
                },
            }
        )
        self.assertEqual(cleaned.get("name"), "Alice")
        self.assertNotIn("verification_state", cleaned)
        self.assertEqual(cleaned["properties"]["source_chunk_ids"], ["chunk-1"])
        self.assertNotIn("verification_state", cleaned["properties"])


class McpVerificationPermissionTests(unittest.TestCase):
    def test_mcp_mutate_denied_without_verification_scope(self):
        token = auth_token_var.set("test-token")
        scopes = auth_scopes_var.set((DEFAULT_SCOPE,))
        principal = auth_principal_var.set("brainpat:demo")
        try:
            with self.assertRaises(McpToolError) as ctx:
                _mutate_fact_sync(
                    "verify",
                    "rel-1",
                    "demo",
                    "doc",
                    "note",
                )
        finally:
            auth_token_var.reset(token)
            auth_scopes_var.reset(scopes)
            auth_principal_var.reset(principal)
        self.assertIn("VERIFICATION_PERMISSION_DENIED", str(ctx.exception))

    def test_mcp_verified_side_denied_without_verification_scope(self):
        from unittest.mock import patch

        token = auth_token_var.set("test-token")
        scopes = auth_scopes_var.set((DEFAULT_SCOPE,))
        principal = auth_principal_var.set("brainpat:demo")
        try:
            with patch("src.services.mcp.main.guard_brainpat", return_value=True):
                with self.assertRaises(McpToolError) as ctx:
                    _resolve_conflict_sync(
                        "demo",
                        "c-1",
                        "rel-1",
                        "note",
                        "verified_side",
                    )
        finally:
            auth_token_var.reset(token)
            auth_scopes_var.reset(scopes)
            auth_principal_var.reset(principal)
        self.assertIn("VERIFICATION_PERMISSION_DENIED", str(ctx.exception))

    def test_search_memory_filters_serialized_rejected_rows(self):
        class _Graph:
            def execute_operation(self, operation, brain_id):
                return {
                    "records": [
                        {"n": {"uuid": "bad", "verification_state": REJECTED}},
                        {"n": {"uuid": "ok", "verification_state": UNVERIFIED}},
                    ]
                }

        from src.adapters.graph import GraphAdapter
        from unittest.mock import patch

        adapter = GraphAdapter()
        adapter.add_client(_Graph())
        token = auth_token_var.set("test-token")
        try:
            with patch("src.services.mcp.main.guard_brainpat", return_value=True), patch(
                "src.services.mcp.main.graph_adapter", adapter
            ):
                payload = _search_memory_sync(
                    "MATCH (n) RETURN n", "demo", False
                )
        finally:
            auth_token_var.reset(token)
        import json

        body = json.loads(payload)
        self.assertEqual(len(body["records"]), 1)
        self.assertEqual(body["records"][0]["n"]["uuid"], "ok")


class Neo4jVerificationWriteTests(unittest.TestCase):
    def test_history_append_uses_list_concat(self):
        client = Neo4jClient.__new__(Neo4jClient)
        calls = []

        class _Driver:
            def execute_query(self, query, parameters_=None, database_=None):
                calls.append((query, parameters_, database_))
                return SimpleNamespace(records=[{"uuid": "rel-1"}])

        client.driver = _Driver()
        client.ensure_database = lambda brain_id: None
        client.get_fact_record = lambda fact_id, brain_id: {
            "kind": "relationship",
            "uuid": fact_id,
            "properties": {},
        }
        client._bind_property_value = lambda value: value
        client._run_query = lambda query, brain_id, params, **kwargs: client.driver.execute_query(
            query, parameters_=params, database_=brain_id
        )
        entry = serialize_history_entry(
            {
                "from_state": UNVERIFIED,
                "to_state": VERIFIED,
                "verified_by": "brainpat:demo",
                "verified_at": "2026-11-18T00:00:00+00:00",
            }
        )
        client.set_fact_verification(
            "rel-1",
            "braina",
            kind="relationship",
            properties={"verification_state": VERIFIED},
            history_entry=entry,
        )
        query, params, _db = calls[0]
        self.assertIn("+ [$history_entry]", query)
        self.assertEqual(params["history_entry"], entry)
        self.assertNotIn("MATCH (n) DETACH DELETE", query)

    def test_extra_props_drop_verification_fields(self):
        client = Neo4jClient.__new__(Neo4jClient)
        extra = client._relationship_property_primitives(
            {
                "verification_state": REJECTED,
                "verified_by": "spoof",
                "v_id": "v1",
            }
        )
        self.assertEqual(extra.get("v_id"), "v1")
        self.assertNotIn("verification_state", extra)
        self.assertNotIn("verified_by", extra)

    def test_citing_query_does_not_dot_access_string_properties(self):
        client = Neo4jClient.__new__(Neo4jClient)
        queries = []

        class _Driver:
            def execute_query(self, query, parameters_=None, database_=None):
                queries.append(query)
                return SimpleNamespace(records=[])

        client.driver = _Driver()
        client.ensure_database = lambda brain_id: None
        client._python_properties = lambda value: value or {}
        client.nodes_and_edges_citing("braina", ["chunk-1"], [])
        node_query, edge_query = queries[0], queries[1]
        self.assertIn("n.properties IS :: MAP", node_query)
        self.assertIn("n.properties IS :: STRING", node_query)
        self.assertIn("r.properties IS :: MAP", edge_query)
        self.assertIn("r.properties IS :: STRING", edge_query)

    def test_add_nodes_skips_cleaned_verification_keys(self):
        client = Neo4jClient.__new__(Neo4jClient)
        calls = []
        client.ensure_database = lambda brain_id: None
        client._bind_property_value = lambda value: value
        client._quoted_labels = lambda labels: "PERSON"
        client._run_query = (
            lambda query, brain_id, params, **kwargs: calls.append(params)
            or SimpleNamespace(records=[])
        )
        node = Node(
            uuid="n1",
            name="Alice",
            labels=["PERSON"],
            properties={"VERIFIED_BY": "spoof", "age": 30},
        )
        client.add_nodes([node], "braina")
        props = calls[0]["props"]
        self.assertEqual(props.get("age"), 30)
        self.assertNotIn("verified_by", props)
        self.assertNotIn("VERIFIED_BY", props)

    def test_networkx_add_nodes_strips_nested_verification_keys(self):
        client = NetworkXGraphClient.__new__(NetworkXGraphClient)
        captured = {}

        def _merge_node(brain_id, labels, identification, properties):
            captured["identification"] = identification
            captured["properties"] = properties
            return "n1"

        client._store = SimpleNamespace(
            ensure_database=lambda brain_id: None,
            merge_node=_merge_node,
        )
        client._clean_property_key = lambda key: key
        node = Node(
            uuid="n1",
            name="Alice",
            labels=["EVENT"],
            properties={
                "verification_state": VERIFIED,
                "properties": {
                    "verification_state": VERIFIED,
                    "source_chunk_ids": ["chunk-1"],
                },
            },
        )
        client.add_nodes(
            [node],
            "braina",
            identification_params={"verification_state": VERIFIED, "alias": "a1"},
        )
        props = captured["properties"]
        identification = captured["identification"]
        self.assertNotIn("verification_state", props)
        self.assertEqual(props["properties"]["source_chunk_ids"], ["chunk-1"])
        self.assertNotIn("verification_state", props["properties"])
        self.assertNotIn("verification_state", identification)
        self.assertEqual(identification.get("alias"), "a1")

    def test_extra_props_drop_cleaned_verification_keys(self):
        client = Neo4jClient.__new__(Neo4jClient)
        extra = client._relationship_property_primitives(
            {
                "VERIFIED_BY": "spoof",
                "Verification_State": REJECTED,
                "v_id": "v1",
            }
        )
        self.assertEqual(extra.get("v_id"), "v1")
        self.assertNotIn("verified_by", extra)
        self.assertNotIn("VERIFIED_BY", extra)
        self.assertNotIn("verification_state", extra)
        self.assertNotIn("Verification_State", extra)


class ServiceIgnoresCallerIdentityWhenActorSet(unittest.TestCase):
    def test_actor_wins_over_verified_by(self):
        class _Graph:
            def __init__(self):
                self.row = {
                    "kind": "relationship",
                    "uuid": "rel-1",
                    "name": "KNOWS",
                    "description": "",
                    "properties": {"uuid": "rel-1"},
                    "labels": [],
                }

            def get_fact_record(self, fact_id, brain_id):
                return self.row

            def set_fact_verification(
                self, fact_id, brain_id, *, kind, properties, history_entry=None
            ):
                from src.core.verification.facts import serialize_history_entry

                props = dict(self.row["properties"])
                props.update(properties)
                entry = serialize_history_entry(history_entry)
                if entry:
                    props["verification_history"] = [entry]
                self.row["properties"] = props
                return self.row

        record = apply_verification_action(
            _Graph(),
            fact_id="rel-1",
            brain_id="demo",
            action="verify",
            actor="brainpat:demo",
            verified_by="attacker",
        )
        self.assertEqual(record.verification.verified_by, "brainpat:demo")


if __name__ == "__main__":
    unittest.main()
