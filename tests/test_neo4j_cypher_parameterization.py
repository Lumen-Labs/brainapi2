import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from neo4j.time import DateTime
from pydantic import ValidationError

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

from src.constants.data import Observation
from src.constants.kg import IdentificationParams, Node, Predicate
from src.lib.cypher_safety import InvalidCypherIdentifier, clean_label
from src.lib.neo4j import client as neo4j_client
from src.lib.neo4j.client import (
    VALIDITY_INDEX_FAIL_BACKOFF_INITIAL_SECONDS,
    VALIDITY_INDEX_REFRESH_TTL_SECONDS,
    _VALIDITY_REL_INDEX_KEYS,
    Neo4jClient,
    reset_validity_index_cache,
)

INJECTION_VALUES = [
    "O'Brien",
    "foo}bar",
    "x`y",
    "a//b",
    "DETACH DELETE",
    "MATCH (n) DETACH DELETE n",
    "名前",
    "'; MATCH (n) DETACH DELETE n //",
]


class _Record(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _DriverRecord:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        raise RuntimeError("neo4j Record.items() must not be required")


class _Result:
    def __init__(self, records=None):
        self.records = records or []


class _Driver:
    def __init__(self, records=None):
        self.calls: list[tuple[str, dict, str | None]] = []
        self._records = records or []

    def execute_query(self, query, parameters_=None, database_=None):
        self.calls.append((query, dict(parameters_ or {}), database_))
        return _Result(self._records)


def _client(records=None) -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.driver = _Driver(records)
    client.ensure_database = MagicMock()
    return client


def _person(uuid="alice-1", name="Alice"):
    return Node(uuid=uuid, name=name, labels=["PERSON"], properties={})


def _assert_value_not_in_query(test, query, value):
    test.assertNotIn(value, query)
    test.assertNotIn(value.replace("'", "\\'"), query)


class Neo4jQueryParameterizationTests(unittest.TestCase):
    def test_add_nodes_parameterizes_uuid_and_properties(self):
        client = _client()
        node = _person()
        node.description = "v1"
        client.add_nodes([node], "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("MERGE (n:`PERSON` {uuid: $uuid})", query)
        self.assertIn("n += $props", query)
        self.assertEqual(params["uuid"], "alice-1")
        self.assertEqual(params["props"]["name"], "Alice")
        self.assertEqual(params["description"], "v1")
        self.assertNotIn("'alice-1'", query)
        self.assertNotIn("'Alice'", query)

    def test_bind_property_value_keeps_homogeneous_lists_and_json_mixed(self):
        client = _client()
        self.assertEqual(client._bind_property_value([1, 2, 3]), [1, 2, 3])
        self.assertEqual(client._bind_property_value(["a", "b"]), ["a", "b"])
        self.assertEqual(client._bind_property_value([]), [])
        mixed_null = client._bind_property_value([1, None])
        mixed_types = client._bind_property_value([1, "two"])
        self.assertIsInstance(mixed_null, str)
        self.assertIsInstance(mixed_types, str)
        self.assertIn("null", mixed_null)
        self.assertIn("two", mixed_types)

    def test_python_properties_convert_neo4j_temporals(self):
        from datetime import datetime as dt
        from neo4j.time import DateTime

        client = _client()
        neo4j_dt = DateTime(2026, 11, 16, 20, 17, 39)
        coerced = client._python_value(neo4j_dt)
        self.assertIsInstance(coerced, dt)
        props = client._python_properties({"last_updated": neo4j_dt, "note": "x"})
        self.assertIsInstance(props["last_updated"], dt)
        self.assertEqual(props["note"], "x")
        dumped = Node(
            uuid="n1",
            name="Ada",
            labels=["PERSON"],
            properties=props,
        ).model_dump(mode="json")
        self.assertEqual(dumped["properties"]["note"], "x")
        self.assertIsInstance(dumped["properties"]["last_updated"], str)

    def test_python_datetime_does_not_invent_now(self):
        from datetime import datetime as dt

        client = _client()
        self.assertIsNone(client._python_datetime(None))
        self.assertIsNone(client._python_datetime(""))
        self.assertIsNone(client._python_datetime("not-a-timestamp"))
        parsed = client._python_datetime("2024-01-02T03:04:05+00:00")
        self.assertIsInstance(parsed, dt)
        self.assertIsNone(client._stored_happened_at("unknown"))
        self.assertIsNone(client._stored_happened_at(""))
        self.assertEqual(
            client._stored_happened_at("2024-01-02T03:04:05+00:00"),
            parsed.isoformat(),
        )

    def test_node_accepts_neo4j_datetime_last_updated(self):
        from datetime import datetime as dt
        from neo4j.time import DateTime

        node = Node(
            uuid="n1",
            name="Ada",
            labels=["PERSON"],
            last_updated=DateTime(2026, 11, 16, 20, 17, 39),
        )
        self.assertIsInstance(node.last_updated, dt)

    def test_event_centric_neighbors_hydrate_neo4j_datetime(self):
        from datetime import datetime as dt
        from neo4j.time import DateTime

        neo4j_dt = DateTime(2026, 11, 18, 15, 32, 53, 3725000)
        client = _client()
        client._run_query = MagicMock(
            return_value=_Result(
                [
                    _Record(
                        {
                            "n_uuid": "alice",
                            "n_name": "Alice",
                            "n_labels": ["PERSON"],
                            "n_last_updated": neo4j_dt,
                            "r_uuid": "made",
                            "r_type": "MADE",
                            "r_direction": "out",
                            "r_last_updated": neo4j_dt,
                            "m_uuid": "event",
                            "m_name": "Meeting",
                            "m_labels": ["EVENT"],
                            "m_last_updated": neo4j_dt,
                            "r2_uuid": "targeted",
                            "r2_type": "TARGETED",
                            "r2_direction": "out",
                            "r2_last_updated": neo4j_dt,
                            "b_uuid": "bob",
                            "b_name": "Bob",
                            "b_labels": ["PERSON"],
                            "b_last_updated": neo4j_dt,
                        }
                    )
                ]
            )
        )
        triples = client.get_event_centric_neighbors(["alice"], "braina")
        self.assertEqual(len(triples), 1)
        for entity in triples[0]:
            self.assertIsInstance(entity.last_updated, dt)
            dumped = entity.model_dump(mode="json")
            self.assertIsInstance(dumped["last_updated"], str)

    def test_add_relationship_returns_ok_string(self):
        client = _client(records=[_Record({"a": 1})])
        predicate = Predicate(uuid="rel-1", name="OWNS", description="owns")
        result = client.add_relationship(
            _person("a"), predicate, _person("b", "Widget"), "braina"
        )
        self.assertEqual(result, "ok")

    def test_add_nodes_serializes_metadata_maps_for_storage(self):
        client = _client()
        node = _person()
        node.metadata = {"source": "api"}
        client.add_nodes([node], "braina", metadata={"origin": "ingest"})
        _query, params, _db = client.driver.calls[0]
        self.assertIsInstance(params["props"]["metadata"], str)
        self.assertIn("source", params["props"]["metadata"])
        self.assertIn("origin", params["props"]["metadata"])

    def test_add_nodes_injection_payload_stays_in_params(self):
        client = _client()
        for payload in INJECTION_VALUES:
            client.driver.calls.clear()
            node = Node(
                uuid="n-1",
                name=payload,
                labels=["PERSON"],
                description=payload,
                properties={"note": payload},
            )
            client.add_nodes([node], "braina")
            query, params, _db = client.driver.calls[0]
            _assert_value_not_in_query(self, query, payload)
            self.assertEqual(params["props"]["name"], payload)
            self.assertEqual(params["description"], payload)
            self.assertEqual(params["props"]["note"], payload)

    def test_add_nodes_keeps_ingest_task_ids_as_list_property(self):
        client = _client()
        node = _person()
        node.properties = {"ingest_task_ids": ["task-1", "task-2"]}
        client.add_nodes([node], "braina")
        _query, params, _db = client.driver.calls[0]
        self.assertEqual(params["ingest_task_ids"], ["task-1", "task-2"])
        self.assertNotIn("ingest_task_ids", params["props"])

    def test_add_nodes_maps_non_ascii_labels_and_skips_invalid(self):
        client = _client()
        added = client.add_nodes(
            [
                Node(uuid="n-city", name="Roma", labels=["CITTÀ"]),
                Node(uuid="n-name", name="Hanako", labels=["名前"]),
                Node(uuid="n-person", name="Ada", labels=["PERSON"]),
                Node(uuid="n-bad", name="Skip", labels=["PERSON`x"]),
            ],
            "braina",
        )
        self.assertEqual([node.uuid for node in added], ["n-city", "n-name", "n-person"])
        queries = [call[0] for call in client.driver.calls]
        self.assertEqual(len(queries), 3)
        self.assertIn("MERGE (n:`CITTA` {uuid: $uuid})", queries[0])
        self.assertIn(f"MERGE (n:`{clean_label('名前')}` {{uuid: $uuid}})", queries[1])
        self.assertIn("MERGE (n:`PERSON` {uuid: $uuid})", queries[2])
        city_params = client.driver.calls[0][1]
        self.assertEqual(city_params["props"]["original_type"], "CITTÀ")
        name_params = client.driver.calls[1][1]
        self.assertEqual(name_params["props"]["original_type"], "名前")
        with self.assertRaises(InvalidCypherIdentifier):
            client.add_nodes(
                [Node(uuid="only-bad", name="Skip", labels=["PERSON`x"])],
                "braina",
            )

    def test_add_relationship_parameterizes_endpoint_and_predicate_uuids(self):
        client = _client(records=[_Record({"a": 1})])
        predicate = Predicate(
            uuid="rel-1",
            name="OWNS",
            description="owns",
            properties={
                "v_id": "v1",
                "valid_from": "2026-01-01T00:00:00Z",
                "source": "structured_deterministic",
            },
            flow_key="flow-1",
        )
        client.add_relationship(_person("a"), predicate, _person("b", "Widget"), "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("WHERE a['uuid'] = $subject_uuid", query)
        self.assertIn("WHERE b['uuid'] = $object_uuid", query)
        self.assertIn("MERGE (a)-[r:`OWNS` {uuid: $predicate_uuid}]->(b)", query)
        self.assertEqual(params["subject_uuid"], "a")
        self.assertEqual(params["object_uuid"], "b")
        self.assertEqual(params["predicate_uuid"], "rel-1")
        extra = params["extra_props"]
        self.assertNotIsInstance(extra.get("properties"), dict)
        self.assertEqual(extra.get("v_id"), "v1")
        self.assertEqual(extra.get("valid_from"), "2026-01-01T00:00:00Z")
        self.assertEqual(extra.get("source"), "structured_deterministic")
        self.assertIsNone(extra.get("last_updated"))

        stamped = Predicate(
            uuid="rel-2",
            name="OWNS",
            description="owns",
            last_updated="2026-01-02T03:04:05+00:00",
            properties={"v_id": "v2"},
        )
        client.add_relationship(_person("a"), stamped, _person("b", "Widget"), "braina")
        stamped_extra = client.driver.calls[1][1]["extra_props"]
        self.assertIsInstance(stamped_extra.get("last_updated"), str)

    def test_add_relationship_keeps_zero_amount_and_omits_endpoint_uuid(self):
        client = _client(records=[_Record({"a": 1})])
        subject = _person("a")
        subject.properties = {"uuid": "node-a", "note": "subject"}
        obj = _person("b", "Widget")
        obj.properties = {"uuid": "node-b"}
        predicate = Predicate(
            uuid="rel-1",
            name="OWNS",
            description="owns",
            amount=0.0,
            properties={"uuid": "should-not-win", "v_id": "v1"},
        )
        client.add_relationship(subject, predicate, obj, "braina")
        extra = client.driver.calls[0][1]["extra_props"]
        self.assertEqual(extra.get("amount"), 0.0)
        self.assertNotIn("uuid", extra)
        self.assertEqual(extra.get("v_id"), "v1")

    def test_deprecate_relationship_does_not_interpolate_names(self):
        client = _client()
        for payload in INJECTION_VALUES:
            client.driver.calls.clear()
            client.deprecate_relationship(
                Node(uuid="a", name=payload, labels=["PERSON"]),
                Predicate(name="KNOWS", description=""),
                Node(uuid="b", name=payload, labels=["PERSON"]),
                "braina",
            )
            query, params, _db = client.driver.calls[0]
            self.assertIn("MATCH (a:`PERSON`) WHERE a['name'] = $subject_name", query)
            self.assertIn("MATCH (a)-[r:`KNOWS`]->(b)", query)
            _assert_value_not_in_query(self, query, payload)
            self.assertEqual(params["subject_name"], payload)
            self.assertEqual(params["object_name"], payload)

    def test_deprecate_relationship_rejects_identifier_injection_in_labels(self):
        client = _client()
        with self.assertRaises(InvalidCypherIdentifier):
            client.deprecate_relationship(
                Node(uuid="a", name="Alice", labels=["PERSON`DETACH DELETE"]),
                Predicate(name="KNOWS", description=""),
                Node(uuid="b", name="Bob", labels=["PERSON"]),
                "braina",
            )
        self.assertEqual(client.driver.calls, [])

    def test_search_graph_uses_unwind_params(self):
        client = _client()
        client.search_graph([_person(name="O'Brien")], "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("UNWIND $nodes AS node", query)
        self.assertNotIn("O'Brien", query)
        self.assertEqual(params["nodes"][0]["name"], "O'Brien")
        self.assertEqual(params["nodes"][0]["labels"], ["PERSON"])

    def test_node_text_search_parameterizes_text(self):
        client = _client()
        client.node_text_search("MATCH (n) DETACH DELETE n", "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("CONTAINS toLower($text)", query)
        self.assertNotIn("DETACH DELETE", query)
        self.assertEqual(params["text"], "MATCH (n) DETACH DELETE n")

    def test_node_text_search_maps_uuid_from_returned_fields(self):
        record = _DriverRecord(
            {
                "uuid": "alice-1",
                "name": "Alice",
                "labels": ["PERSON"],
                "description": "",
                "properties": {"uuid": "alice-1", "name": "Alice"},
            }
        )
        client = _client(records=[record])
        nodes = client.node_text_search("Alice", "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("n['uuid'] as uuid", query)
        self.assertIn("n['name'] as name", query)
        self.assertNotRegex(query, r"RETURN n\s*$")
        self.assertEqual(params["text"], "Alice")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uuid, "alice-1")
        self.assertEqual(nodes[0].name, "Alice")

    def test_get_nodes_by_uuid_parameterizes_ids_and_bounds_depth(self):
        client = _client()
        client.get_nodes_by_uuid(
            ["u-1", "'; MATCH (n) DETACH DELETE n //"],
            "braina",
            with_relationships=True,
            relationships_depth=2,
            relationships_type=["KNOWS"],
            preferred_labels=["PERSON"],
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("n['uuid'] IN $uuids", query)
        self.assertIn("OPTIONAL MATCH (n)-[r:`KNOWS`*1..2]-(m)", query)
        self.assertIn("lbl IN $preferred_labels", query)
        self.assertEqual(params["uuids"][1], "'; MATCH (n) DETACH DELETE n //")
        self.assertNotIn("DETACH DELETE", query)
        with self.assertRaises(InvalidCypherIdentifier):
            client.get_nodes_by_uuid(["u-1"], "braina", with_relationships=True, relationships_depth="2]-(x)")

    def test_get_by_uuids_and_neighbors_and_connected_nodes(self):
        client = _client()
        payload = "'; MATCH (n) DETACH DELETE n //"
        client.get_by_uuids([payload], "braina")
        client.get_neighbors([payload], "braina", of_types=["PERSON"], limit=5)
        client.get_neighbor_node_tuples(payload, [payload], "braina")
        client.get_connected_nodes("braina", uuids=[payload], with_labels=["PERSON"])
        for query, params, _db in client.driver.calls:
            self.assertNotIn(payload, query)
            flattened = str(params)
            self.assertIn(payload, flattened)

    def test_search_entities_json_serializes_datetime_properties(self):
        from datetime import datetime, timezone

        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        client = _client(
            records=[
                {
                    "uuid": "n1",
                    "name": "Alice",
                    "labels": ["PERSON"],
                    "description": "",
                    "properties": {"last_updated": stamp, "name": "Alice"},
                }
            ]
        )
        result = client.search_entities("braina", limit=1)
        entity = result.results[0]
        self.assertIsInstance(entity.properties["last_updated"], str)
        dumped = entity.model_dump(mode="json")
        self.assertIsInstance(dumped["properties"]["last_updated"], str)

    def test_search_entities_and_relationships_parameterize_query_text(self):
        client = _client()
        payload = "O'Brien'; MATCH (n) DETACH DELETE n"
        client.search_entities("braina", limit=3, skip=1, node_labels=["PERSON"], query_text=payload)
        client.search_relationships(
            "braina",
            limit=3,
            skip=1,
            relationship_types=["KNOWS"],
            from_node_labels=["PERSON"],
            to_node_labels=["ORG"],
            relationship_uuids=[payload],
            query_text=payload,
        )
        for query, params, _db in client.driver.calls:
            self.assertNotIn(payload, query)
            self.assertEqual(params["query_text"], payload)
        self.assertTrue(any("SKIP $skip" in query for query, _p, _d in client.driver.calls))
        self.assertTrue(any("LIMIT $limit" in query for query, _p, _d in client.driver.calls))
        relationship_queries = [
            query for query, _p, _d in client.driver.calls if "MATCH (n)-[r]->(m)" in query
        ]
        self.assertGreaterEqual(len(relationship_queries), 2)
        self.assertTrue(all("MATCH (n)-[r]->(m)" in query for query in relationship_queries))

    def test_update_properties_uses_dynamic_keys(self):
        client = _client()
        payload = "MATCH (n) DETACH DELETE n"
        client.update_properties(
            "node-1",
            "node",
            "braina",
            {"note": payload},
            ["stale"],
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("t[$set_key_0] = $set_val_0", query)
        self.assertIn("t[$rm_key_0] = null", query)
        self.assertEqual(params["set_val_0"], payload)
        self.assertNotIn(payload, query)

    def test_update_node_parameterizes_values_and_quotes_labels(self):
        records = [
            _Record(
                {
                    "uuid": "node-1",
                    "name": "Alice",
                    "labels": ["PERSON"],
                    "description": "",
                    "properties": {},
                }
            )
        ]
        client = _client(records)
        payload = "O'Brien"
        client.update_node(
            "node-1",
            "braina",
            new_name=payload,
            new_description=payload,
            new_labels=["PERSON"],
            new_properties={"note": payload, "meta": {"k": 1}},
            properties_to_remove=["stale"],
        )
        query, params, _db = client.driver.calls[-1]
        self.assertIn("n['name'] = $new_name", query)
        self.assertIn("SET n:`PERSON`", query)
        self.assertEqual(params["new_name"], payload)
        self.assertNotIn(payload, query)
        self.assertIn('{"k": 1}', list(params.values()))

    def test_check_node_existence_quotes_labels(self):
        client = _client()
        client.check_node_existence("u1", "O'Brien", ["PERSON"], "braina")
        query, params, _db = client.driver.calls[0]
        self.assertIn("MATCH (n:`PERSON`)", query)
        self.assertEqual(params["name"], "O'Brien")
        self.assertNotIn("O'Brien", query)

    def test_get_by_identification_params_uses_dynamic_property_keys(self):
        client = _client()
        params_model = IdentificationParams(name="O'Brien")
        client.get_by_identification_params(params_model, "braina", entity_types=["PERSON"])
        query, params, _db = client.driver.calls[0]
        self.assertIn("MATCH (n) ", query)
        self.assertNotIn("MATCH (n:`PERSON`)", query)
        self.assertIn("ANY(lbl IN labels(n) WHERE lbl IN $entity_types)", query)
        self.assertEqual(params["entity_types"], ["PERSON"])
        self.assertIn("n[$key_0] = $param_0", query)
        self.assertIn("O'Brien", params.values())
        self.assertNotIn("O'Brien", query)

    def test_event_centric_triples_removes_and_lists_use_params(self):
        client = _client()
        payload = "'; MATCH (n) DETACH DELETE n //"
        client.get_event_centric_neighbors([payload], "braina")
        client.get_triples_by_uuid([payload], "braina")
        client.remove_nodes([payload], "braina")
        client.list_relationships(payload, payload, "braina")
        client.get_event_hub_facts([payload], "braina")
        client.remove_relationships(
            [
                (
                    {"uuid": payload},
                    {"uuid": payload},
                    {"uuid": payload},
                )
            ],
            "braina",
        )
        client.remove_relationships(
            [
                (
                    {"name": payload, "labels": ["PERSON"]},
                    {"name": "KNOWS"},
                    {"name": payload, "labels": ["PERSON"]},
                )
            ],
            "braina",
        )
        self.assertIn(
            "MATCH (n)-[r:`KNOWS`]-(m)",
            client.driver.calls[-1][0],
        )
        for query, params, _db in client.driver.calls:
            self.assertNotIn(payload, query)
            self.assertTrue(params)

    def test_ensure_database_quotes_name_and_rejects_injection(self):
        client = Neo4jClient.__new__(Neo4jClient)
        client.driver = _Driver()
        client._verify_database_accessible = MagicMock(side_effect=[False, True])
        client.ensure_database("braina")
        query, _params, database = client.driver.calls[0]
        self.assertEqual(query, "CREATE DATABASE `braina`")
        self.assertEqual(database, "system")
        with self.assertRaises(InvalidCypherIdentifier):
            client.ensure_database("brain-a; MATCH (n) DETACH DELETE n")

    def test_list_validity_relationships_parameterizes_subject_uuids(self):
        client = _client()
        payload = "'; MATCH (n) DETACH DELETE n //"
        client.list_validity_relationships(
            "braina",
            skip=2,
            limit=5,
            subject_uuids=[payload],
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("n['uuid'] IN $subject_uuids", query)
        self.assertIn("SKIP $skip LIMIT $limit", query)
        self.assertEqual(params["subject_uuids"], [payload])
        self.assertEqual(params["skip"], 2)
        self.assertEqual(params["limit"], 5)
        self.assertNotIn(payload, query)

    def test_list_validity_relationships_pushes_window_into_cypher(self):
        client = _client()
        client.list_validity_relationships(
            "braina",
            skip=10,
            limit=5,
            since="2024-01-01T00:00:00Z",
            until="2025-01-01T00:00:00Z",
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("WHERE", query)
        self.assertIn("$since", query)
        self.assertIn("$until", query)
        self.assertIn(">= $since", query)
        self.assertIn("< $until", query)
        self.assertIn("SKIP $skip LIMIT $limit", query)
        self.assertNotIn("MATCH (n)-[r]->(m)\nRETURN", query)
        self.assertEqual(params["since"], "2024-01-01T00:00:00Z")
        self.assertEqual(params["until"], "2025-01-01T00:00:00Z")
        self.assertEqual(params["skip"], 10)
        self.assertEqual(params["limit"], 5)
        count_query, count_params, _count_db = client.driver.calls[1]
        self.assertIn("WHERE", count_query)
        self.assertIn("$since", count_query)
        self.assertIn("until", count_params)
        self.assertEqual(count_params["since"], "2024-01-01T00:00:00Z")

    def test_get_event_centric_neighbors_as_of_uses_valid_at_helper(self):
        client = _client()
        client.get_event_centric_neighbors(
            ["alice"], "braina", as_of="2024-06-01T00:00:00Z"
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("$as_of", query)
        self.assertIn("valid_from", query)
        self.assertIn("valid_at", query)
        self.assertEqual(params["as_of"], "2024-06-01T00:00:00Z")

    def test_ensure_database_creates_validity_range_indexes(self):
        reset_validity_index_cache()

        class _TypedDriver(_Driver):
            def execute_query(self, query, parameters_=None, database_=None):
                self.calls.append((query, dict(parameters_ or {}), database_))
                if "relationshipTypes" in query:
                    return _Result([_Record({"relationshipType": "REQUIRES"})])
                return _Result(self._records)

        client = Neo4jClient.__new__(Neo4jClient)
        client.driver = _TypedDriver()
        client._verify_database_accessible = MagicMock(return_value=True)
        client.ensure_database("braina")
        queries = [query for query, _params, _db in client.driver.calls]
        self.assertTrue(any("relationshipTypes" in query for query in queries))
        index_queries = [query for query in queries if "CREATE RANGE INDEX" in query]
        self.assertTrue(index_queries)
        self.assertTrue(
            all("FOR ()-[r:`REQUIRES`]-()" in query for query in index_queries),
            index_queries,
        )
        joined = "\n".join(index_queries)
        for key in (
            "valid_from",
            "valid_to",
            "recorded_at",
            "valid_at",
            "inserted_at",
            "invalid_at",
            "superseded_at",
        ):
            self.assertIn(key, joined)
            self.assertIn("rel_REQUIRES_" + key, joined)

    def test_list_validity_relationships_orders_by_subject_and_predicate(self):
        client = _client()
        client.list_validity_relationships("braina", skip=0, limit=5)
        query, _params, _db = client.driver.calls[0]
        self.assertIn("ORDER BY n['uuid'], pred_key, r['uuid'], elementId(r)", query)
        self.assertIn("=~ '[A-Z0-9]'", query)
        self.assertIn("ELSE '_' END", query)
        self.assertIn("replace(t, '__', '_')", query)
        self.assertIn("AS pred_key", query)
        self.assertNotIn("n['uuid'] >= $after_subject_uuid", query)
        self.assertNotIn("ORDER BY n['uuid'], type(r), r['uuid'], elementId(r)", query)
        self.assertNotIn("ORDER BY r['uuid'], elementId(r)\n", query)

    def test_list_validity_relationships_parameterizes_after_cursor(self):
        client = _client()
        payload = "'; MATCH (n) DETACH DELETE n //"
        client.list_validity_relationships(
            "braina",
            skip=0,
            limit=5,
            after_subject_uuid=payload,
            after_predicate="HAS_WINDOW",
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("$after_subject_uuid", query)
        self.assertIn("$after_predicate", query)
        self.assertIn("pred_key > $after_predicate", query)
        self.assertIn("n['uuid'] >= $after_subject_uuid", query)
        self.assertLess(
            query.index("n['uuid'] >= $after_subject_uuid"),
            query.index("AS pred_key"),
        )
        self.assertNotIn("type(r) > $after_predicate", query)
        self.assertEqual(params["after_subject_uuid"], payload)
        self.assertEqual(params["after_predicate"], "HAS_WINDOW")
        self.assertNotIn(payload, query)
        count_query, count_params, _count_db = client.driver.calls[1]
        self.assertNotIn("$after_subject_uuid", count_query)
        self.assertNotIn("after_subject_uuid", count_params)
        self.assertNotIn("n['uuid'] >= $after_subject_uuid", count_query)

    def test_list_validity_relationships_normalizes_after_predicate_aliases(self):
        client = _client()
        client.list_validity_relationships(
            "braina",
            skip=0,
            limit=5,
            after_subject_uuid="subject-1",
            after_predicate="HAS__A",
        )
        query, params, _db = client.driver.calls[0]
        self.assertEqual(params["after_predicate"], "HAS_A")
        self.assertIn("pred_key > $after_predicate", query)

    def test_list_validity_relationships_normalizes_punctuated_after_predicate(self):
        client = _client()
        client.list_validity_relationships(
            "braina",
            skip=0,
            limit=5,
            after_subject_uuid="subject-1",
            after_predicate="HAS-A",
        )
        query, params, _db = client.driver.calls[0]
        self.assertEqual(params["after_predicate"], "HAS_A")
        self.assertIn("n['uuid'] >= $after_subject_uuid", query)
        self.assertIn("=~ '[A-Z0-9]'", query)

    def test_list_validity_relationships_parameterizes_relationship_uuids(self):
        client = _client()
        payload = "'; MATCH (n) DETACH DELETE n //"
        client.list_validity_relationships(
            "braina",
            skip=1,
            limit=4,
            relationship_uuids=[payload],
        )
        query, params, _db = client.driver.calls[0]
        self.assertIn("r['uuid'] IN $relationship_uuids", query)
        self.assertEqual(params["relationship_uuids"], [payload])
        self.assertNotIn(payload, query)

    def test_execute_operation_rejects_injected_database_name(self):
        client = _client()
        with self.assertRaises(InvalidCypherIdentifier):
            client.execute_operation("RETURN 1", "neo4j`; MATCH (n) DETACH DELETE n")

    def test_add_relationship_requires_matched_endpoints(self):
        subject = _person()
        obj = _person(uuid="bob-1", name="Bob")
        pred = Predicate(uuid="rel-1", name="KNOWS", description="")
        ok_client = _client(records=[_Record({"a": 1})])
        self.assertEqual(
            ok_client.add_relationship(subject, pred, obj, "braina"),
            "ok",
        )
        missing = _client(records=[])
        with self.assertRaises(ValueError):
            missing.add_relationship(subject, pred, obj, "braina")

    def test_remove_nodes_decodes_json_observations_and_temporals(self):
        from datetime import datetime as dt
        from neo4j.time import DateTime

        observations = json.dumps([{"text": "seen", "resource_id": "res-1"}])
        neo4j_dt = DateTime(2026, 11, 18, 12, 0, 0)
        payload = {
            "uuid": "node-1",
            "name": "Alice",
            "labels": ["PERSON"],
            "description": "person",
            "properties": {
                "uuid": "node-1",
                "name": "Alice",
                "note": "ok",
                "observations": observations,
                "last_updated": neo4j_dt,
            },
            "polarity": "neutral",
            "metadata": {"source": "ingest"},
            "happened_at": None,
            "last_updated": neo4j_dt,
            "observations": observations,
        }
        client = _client(records=[_Record({"node": payload})])
        removed = client.remove_nodes(["node-1"], "braina")
        self.assertEqual(len(removed), 1)
        node = removed[0]
        self.assertEqual(node.uuid, "node-1")
        self.assertEqual(node.name, "Alice")
        self.assertEqual(node.observations[0].text, "seen")
        self.assertEqual(node.observations[0].resource_id, "res-1")
        self.assertIsInstance(node.last_updated, dt)
        self.assertEqual(node.properties["note"], "ok")
        query, params, _db = client.driver.calls[0]
        self.assertIn("DELETE n", query)
        self.assertEqual(params["uuids"], ["node-1"])

    def test_list_incident_relationship_uuids_uses_params(self):
        payload = "'; MATCH (n) DETACH DELETE n //"
        client = _client(records=[_Record({"uuid": payload, "rels": [payload, None]})])
        found = client.list_incident_relationship_uuids([payload, "other"], "braina")
        self.assertEqual(found[payload], [payload])
        self.assertEqual(found["other"], [])
        query, params, _db = client.driver.calls[0]
        self.assertNotIn(payload, query)
        self.assertEqual(params["uuids"], [payload, "other"])

    def test_add_nodes_skips_uuid_property_and_encodes_observations(self):
        client = _client(records=[_Record({"n": 1})])
        obs = Observation(text="seen", resource_id="res-1")
        node = Node(
            uuid="node-1",
            name="Alice",
            labels=["PERSON"],
            properties={"uuid": "attacker", "note": "ok"},
            observations=[obs],
        )
        client.add_nodes([node], "braina")
        _query, params, _db = client.driver.calls[0]
        self.assertEqual(params["uuid"], "node-1")
        self.assertNotIn("uuid", params["props"])
        self.assertEqual(params["props"]["note"], "ok")
        self.assertIsInstance(params["props"]["observations"], str)
        decoded = client._decode_observations(params["props"]["observations"])
        self.assertEqual(decoded[0]["text"], "seen")
        self.assertEqual(client._decode_observations("[]"), [])
        self.assertEqual(client._decode_observations(""), [])
        self.assertIsNone(client._decode_observations(None))

    def test_remove_relationships_requires_endpoint_identity(self):
        client = _client()
        with self.assertRaises(InvalidCypherIdentifier):
            client.remove_relationships(
                [({}, {"name": "KNOWS"}, {"uuid": "b"})],
                "braina",
            )
        with self.assertRaises(InvalidCypherIdentifier):
            client.remove_relationships(
                [
                    (
                        {"labels": ["PERSON"]},
                        {"name": "KNOWS"},
                        {"name": "Bob"},
                    )
                ],
                "braina",
            )

    def test_remove_relationships_rejects_mixed_batch_before_uuid_deletes(self):
        client = _client(records=[_Record({"n_uuid": "a"})])
        with self.assertRaises(InvalidCypherIdentifier):
            client.remove_relationships(
                [
                    ({"uuid": "a"}, {"uuid": "rel-1"}, {"uuid": "b"}),
                    ({}, {"name": "KNOWS"}, {"uuid": "b"}),
                ],
                "braina",
            )
        self.assertEqual(client.driver.calls, [])

    def test_get_neighbors_applies_zero_limit(self):
        client = _client()
        client.get_neighbors(["u-1"], "braina", limit=0)
        query, params, _db = client.driver.calls[0]
        self.assertIn("LIMIT $limit", query)
        self.assertEqual(params["limit"], 0)

    def test_fact_verification_reads_and_writes_use_params(self):
        payload = "'; MATCH (n) DETACH DELETE n //"
        client = _client(records=[_Record({"uuid": payload, "kind": "relationship"})])
        client.get_fact_record(payload, "braina")
        client.set_fact_verification(
            payload,
            "braina",
            kind="relationship",
            properties={
                "verification_state": "verified",
                "verified_by": payload,
                "verified_at": "2026-11-16T00:00:00+00:00",
                "verification_source": payload,
                "verification_note": payload,
                "verification_history": [payload],
            },
        )
        client.list_facts_by_verification(
            "braina",
            state="verified",
            include_rejected=False,
            limit=10,
            skip=0,
        )
        self.assertTrue(client.driver.calls)
        for query, params, _db in client.driver.calls:
            self.assertNotIn(payload, query)
            self.assertTrue(params)
        first_query, first_params, _db = client.driver.calls[0]
        self.assertIn("$uuid", first_query)
        self.assertEqual(first_params["uuid"], payload)
        write_query = client.driver.calls[1][0]
        write_params = client.driver.calls[1][1]
        self.assertIn("$verified_by", write_query)
        self.assertEqual(write_params["verified_by"], payload)
        self.assertEqual(write_params["verification_history"], [payload])

    def test_remove_nodes_decodes_json_observations_and_neo4j_datetimes(self):
        stored_at = DateTime(2026, 1, 2, 3, 4, 5, 0)
        observations = json.dumps(
            [
                {
                    "id": "obs-1",
                    "text": "seen after #120",
                    "resource_id": "res-1",
                    "inserted_at": "2026-01-02T03:04:05+00:00",
                }
            ]
        )
        raw_node = {
            "uuid": "n-1",
            "name": "Alice",
            "labels": ["PERSON"],
            "description": "a person",
            "properties": {
                "name": "Alice",
                "last_updated": stored_at,
                "observations": observations,
            },
            "polarity": "neutral",
            "metadata": json.dumps({"source": "purge"}),
            "happened_at": stored_at,
            "last_updated": stored_at,
            "observations": observations,
        }
        with self.assertRaises(ValidationError):
            Node(**raw_node)
        client = _client(records=[_Record({"node": dict(raw_node)})])
        removed = client.remove_nodes(["n-1"], "braina")
        self.assertEqual(len(removed), 1)
        node = removed[0]
        self.assertEqual(node.uuid, "n-1")
        self.assertEqual(node.name, "Alice")
        self.assertEqual(node.labels, ["PERSON"])
        self.assertEqual(node.observations[0].text, "seen after #120")
        self.assertEqual(node.observations[0].resource_id, "res-1")
        self.assertEqual(node.last_updated, stored_at.to_native())
        self.assertTrue(node.happened_at.startswith("2026-01-02T03:04:05"))
        self.assertEqual(node.metadata["source"], "purge")
        self.assertEqual(node.properties["last_updated"], stored_at.to_native())
        query, params, database = client.driver.calls[0]
        self.assertIn("WHERE n['uuid'] IN $uuids", query)
        self.assertEqual(params["uuids"], ["n-1"])
        self.assertEqual(database, "braina")

    def test_get_by_uuids_decodes_json_observations_and_neo4j_datetimes(self):
        stored_at = DateTime(2026, 3, 4, 5, 6, 7, 0)
        observations = json.dumps(
            [
                {
                    "id": "obs-2",
                    "text": "read path",
                    "resource_id": "res-2",
                    "inserted_at": "2026-03-04T05:06:07+00:00",
                }
            ]
        )
        record = _Record(
            {
                "uuid": "n-2",
                "name": "Bob",
                "labels": ["PERSON"],
                "description": "",
                "properties": {"note": "ok", "happened_at": stored_at},
                "polarity": "positive",
                "happened_at": stored_at,
                "last_updated": stored_at,
                "observations": observations,
                "metadata": {"kind": "entity"},
            }
        )
        client = _client(records=[record])
        nodes = client.get_by_uuids(["n-2"], "braina")
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.uuid, "n-2")
        self.assertEqual(node.polarity, "positive")
        self.assertEqual(node.observations[0].text, "read path")
        self.assertEqual(node.last_updated, stored_at.to_native())
        self.assertTrue(node.happened_at.startswith("2026-03-04T05:06:07"))
        self.assertEqual(node.properties["happened_at"], stored_at.to_native())
        self.assertEqual(node.metadata["kind"], "entity")

    def test_node_from_driver_record_does_not_require_items(self):
        stored_at = DateTime(2026, 5, 6, 7, 8, 9, 0)
        observations = json.dumps(
            [
                {
                    "id": "obs-3",
                    "text": "driver record",
                    "resource_id": "res-3",
                    "inserted_at": "2026-05-06T07:08:09+00:00",
                }
            ]
        )
        record = _DriverRecord(
            {
                "uuid": "n-3",
                "name": "Cara",
                "labels": ["PERSON"],
                "description": "",
                "properties": {"note": "ok"},
                "polarity": "neutral",
                "happened_at": stored_at,
                "last_updated": stored_at,
                "observations": observations,
                "metadata": {"kind": "entity"},
            }
        )
        client = _client(records=[record])
        nodes = client.get_by_uuids(["n-3"], "braina")
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.uuid, "n-3")
        self.assertEqual(node.name, "Cara")
        self.assertEqual(node.observations[0].text, "driver record")
        self.assertEqual(node.last_updated, stored_at.to_native())


def _count_calls(calls, needle: str) -> int:
    return sum(1 for query, _params, _db in calls if needle in query)


class _TypedDriver(_Driver):
    def __init__(self, types=None, records=None, fail_index=False):
        super().__init__(records)
        self.types = list(types or [])
        self.fail_index = fail_index

    def execute_query(self, query, parameters_=None, database_=None):
        self.calls.append((query, dict(parameters_ or {}), database_))
        if "relationshipTypes" in query:
            return _Result(
                [_Record({"relationshipType": rel_type}) for rel_type in self.types]
            )
        if self.fail_index and "CREATE RANGE INDEX" in query:
            raise RuntimeError("index creation failed")
        return _Result(self._records)


class Neo4jValidityIndexCacheTests(unittest.TestCase):
    def setUp(self):
        reset_validity_index_cache()
        self._clock = 1_000.0
        self._orig_now = neo4j_client._validity_index_now
        neo4j_client._validity_index_now = lambda: self._clock

    def tearDown(self):
        neo4j_client._validity_index_now = self._orig_now
        reset_validity_index_cache()

    def _client(self, types=None, records=None, fail_index=False) -> Neo4jClient:
        client = Neo4jClient.__new__(Neo4jClient)
        client.driver = _TypedDriver(
            types=types, records=records, fail_index=fail_index
        )
        client._verify_database_accessible = MagicMock(return_value=True)
        return client

    def test_second_ensure_database_skips_type_listing_and_index_creates(self):
        client = self._client(types=["REQUIRES"])
        client.ensure_database("braina")
        first_lists = _count_calls(client.driver.calls, "relationshipTypes")
        first_indexes = _count_calls(client.driver.calls, "CREATE RANGE INDEX")
        self.assertEqual(first_lists, 1)
        self.assertEqual(first_indexes, len(_VALIDITY_REL_INDEX_KEYS))

        client.ensure_database("braina")
        self.assertEqual(
            _count_calls(client.driver.calls, "relationshipTypes"), first_lists
        )
        self.assertEqual(
            _count_calls(client.driver.calls, "CREATE RANGE INDEX"), first_indexes
        )

    def test_ttl_expiry_relists_types_but_skips_already_indexed_creates(self):
        client = self._client(types=["REQUIRES"])
        client.ensure_database("braina")
        self._clock += VALIDITY_INDEX_REFRESH_TTL_SECONDS + 1
        client.ensure_database("braina")
        self.assertEqual(_count_calls(client.driver.calls, "relationshipTypes"), 2)
        self.assertEqual(
            _count_calls(client.driver.calls, "CREATE RANGE INDEX"),
            len(_VALIDITY_REL_INDEX_KEYS),
        )

    def test_new_type_write_creates_indexes_without_relisting(self):
        client = self._client(
            types=["REQUIRES"], records=[_Record({"a": 1})]
        )
        client.ensure_database("braina")
        lists_after_ensure = _count_calls(client.driver.calls, "relationshipTypes")
        indexes_after_ensure = _count_calls(
            client.driver.calls, "CREATE RANGE INDEX"
        )
        self.assertEqual(lists_after_ensure, 1)

        predicate = Predicate(uuid="rel-1", name="KNOWS", description="")
        client.add_relationship(_person("a"), predicate, _person("b", "Bob"), "braina")
        self.assertEqual(
            _count_calls(client.driver.calls, "relationshipTypes"),
            lists_after_ensure,
        )
        self.assertEqual(
            _count_calls(client.driver.calls, "CREATE RANGE INDEX"),
            indexes_after_ensure + len(_VALIDITY_REL_INDEX_KEYS),
        )
        knows_indexes = [
            query
            for query, _params, _db in client.driver.calls
            if "CREATE RANGE INDEX" in query and "r:`KNOWS`" in query
        ]
        self.assertEqual(len(knows_indexes), len(_VALIDITY_REL_INDEX_KEYS))

    def test_failed_index_create_is_cached_with_backoff_and_logged_once(self):
        client = self._client(types=["REQUIRES"], fail_index=True)
        with patch.object(neo4j_client, "logger") as mock_logger:
            client.ensure_database("braina")
            client.ensure_database("braina")
            self.assertEqual(_count_calls(client.driver.calls, "relationshipTypes"), 1)
            self.assertEqual(
                _count_calls(client.driver.calls, "CREATE RANGE INDEX"),
                len(_VALIDITY_REL_INDEX_KEYS),
            )
            self.assertEqual(mock_logger.warning.call_count, 1)

            self._clock += VALIDITY_INDEX_FAIL_BACKOFF_INITIAL_SECONDS + 1
            client.ensure_database("braina")
            self.assertEqual(
                _count_calls(client.driver.calls, "CREATE RANGE INDEX"),
                2 * len(_VALIDITY_REL_INDEX_KEYS),
            )
            self.assertEqual(mock_logger.warning.call_count, 1)

    def test_cache_is_process_level_and_per_database(self):
        first = self._client(types=["REQUIRES"])
        first.ensure_database("braina")
        second = self._client(types=["REQUIRES"])
        second.ensure_database("braina")
        self.assertEqual(_count_calls(second.driver.calls, "relationshipTypes"), 0)
        self.assertEqual(_count_calls(second.driver.calls, "CREATE RANGE INDEX"), 0)

        other = self._client(types=["KNOWS"])
        other.ensure_database("brainb")
        self.assertEqual(_count_calls(other.driver.calls, "relationshipTypes"), 1)
        self.assertEqual(
            _count_calls(other.driver.calls, "CREATE RANGE INDEX"),
            len(_VALIDITY_REL_INDEX_KEYS),
        )
        knows_indexes = [
            query
            for query, _params, _db in other.driver.calls
            if "CREATE RANGE INDEX" in query and "r:`KNOWS`" in query
        ]
        self.assertEqual(len(knows_indexes), len(_VALIDITY_REL_INDEX_KEYS))

    def test_warm_cache_is_thread_safe_and_does_not_relist(self):
        client = self._client(types=["REQUIRES"])
        client.ensure_database("braina")
        before_lists = _count_calls(client.driver.calls, "relationshipTypes")
        before_indexes = _count_calls(client.driver.calls, "CREATE RANGE INDEX")
        errors: list[BaseException] = []

        def _worker():
            try:
                for _ in range(20):
                    client.ensure_database("braina")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(
            _count_calls(client.driver.calls, "relationshipTypes"), before_lists
        )
        self.assertEqual(
            _count_calls(client.driver.calls, "CREATE RANGE INDEX"), before_indexes
        )


if __name__ == "__main__":
    unittest.main()
