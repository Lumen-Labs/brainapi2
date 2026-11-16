import os
import unittest
from unittest.mock import MagicMock

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

from src.constants.kg import IdentificationParams, Node, Predicate
from src.lib.cypher_safety import InvalidCypherIdentifier
from src.lib.neo4j.client import Neo4jClient

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

    def test_add_relationship_parameterizes_endpoint_and_predicate_uuids(self):
        client = _client()
        predicate = Predicate(
            uuid="rel-1",
            name="OWNS",
            description="owns",
            properties={"v_id": "v1"},
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
            new_properties={"note": payload},
            properties_to_remove=["stale"],
        )
        query, params, _db = client.driver.calls[-1]
        self.assertIn("n['name'] = $new_name", query)
        self.assertIn("SET n:`PERSON`", query)
        self.assertEqual(params["new_name"], payload)
        self.assertNotIn(payload, query)

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
        self.assertIn("MATCH (n:`PERSON`)", query)
        self.assertIn("n[$key_0] = $param_0", query)
        self.assertEqual(params["param_0"], "O'Brien")
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

    def test_execute_operation_rejects_injected_database_name(self):
        client = _client()
        with self.assertRaises(InvalidCypherIdentifier):
            client.execute_operation("RETURN 1", "neo4j`; MATCH (n) DETACH DELETE n")


if __name__ == "__main__":
    unittest.main()
