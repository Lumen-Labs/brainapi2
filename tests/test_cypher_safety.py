import os
import unittest

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

from src.lib.cypher_safety import (
    InvalidCypherIdentifier,
    RawCypherRejected,
    clean_label,
    clean_labels,
    clean_property_key,
    create_database_if_not_exists_query,
    create_database_query,
    quote_database_name,
    quote_identifier,
    quote_label,
    quote_labels,
    quote_rel_type,
    quote_rel_types_pattern,
    relationship_changed_in_window_clause,
    relationship_valid_at_clause,
    require_database_name,
    require_int,
    require_limit,
    show_database_status_query,
    static_property_key,
    validate_raw_cypher,
)


class CypherSafetyHelperTests(unittest.TestCase):
    def test_clean_label_preserves_historical_normalization(self):
        self.assertEqual(clean_label("Person"), "PERSON")
        self.assertEqual(clean_label("Person Type"), "PERSON_TYPE")
        self.assertEqual(clean_label("foo-bar"), "FOO_BAR")
        self.assertEqual(clean_labels(["Person", "Org"]), ["PERSON", "ORG"])

    def test_quote_identifier_backticks_valid_names(self):
        self.assertEqual(quote_identifier("PERSON"), "`PERSON`")
        self.assertEqual(quote_label("Person"), "`PERSON`")
        self.assertEqual(quote_labels(["Person", "Event"]), "`PERSON`:`EVENT`")
        self.assertEqual(quote_rel_type("owns"), "`OWNS`")
        self.assertEqual(
            quote_rel_types_pattern(["knows", "likes"]), "`KNOWS`|`LIKES`"
        )

    def test_rejects_injection_shaped_identifiers(self):
        payloads = [
            "x`y",
            "a//b",
            "名前",
            "'; MATCH (n) DETACH DELETE n //",
            "foo@DETACH DELETE n",
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidCypherIdentifier):
                    quote_label(payload)

    def test_property_key_camel_cases_spaces_like_before(self):
        self.assertEqual(clean_property_key("Source Chunk Ids"), "sourceChunkIds")
        self.assertEqual(clean_property_key("uuid"), "uuid")

    def test_static_property_key_allowlist(self):
        self.assertEqual(static_property_key("description"), "description")
        self.assertEqual(static_property_key("ingest_task_ids"), "ingest_task_ids")
        with self.assertRaises(InvalidCypherIdentifier):
            static_property_key("not_allowlisted")

    def test_database_name_matches_existing_brain_id_rules(self):
        self.assertEqual(require_database_name("default"), "default")
        self.assertEqual(require_database_name("locomoconv26"), "locomoconv26")
        self.assertEqual(quote_database_name("neo4j"), "`neo4j`")
        self.assertEqual(
            create_database_query("demo"), "CREATE DATABASE `demo`"
        )
        self.assertEqual(
            create_database_if_not_exists_query("demo"),
            "CREATE DATABASE `demo` IF NOT EXISTS",
        )
        self.assertEqual(
            show_database_status_query("demo"),
            "SHOW DATABASE `demo` YIELD currentStatus",
        )
        self.assertEqual(quote_database_name("brain-id"), "`brain-id`")
        with self.assertRaises(InvalidCypherIdentifier):
            require_database_name("system")
        with self.assertRaises(InvalidCypherIdentifier):
            require_database_name("brain; MATCH (n) DETACH DELETE n")
        with self.assertRaises(InvalidCypherIdentifier):
            require_database_name("1abc")
        with self.assertRaises(InvalidCypherIdentifier):
            require_database_name("foo`bar")

    def test_validity_clauses_use_allowlisted_keys_and_parameters(self):
        valid_at = relationship_valid_at_clause("r", "as_of")
        self.assertIn("r['valid_from']", valid_at)
        self.assertIn("r['valid_to']", valid_at)
        self.assertIn("$as_of", valid_at)
        self.assertNotIn(";", valid_at)
        window = relationship_changed_in_window_clause("r", "since", "until")
        self.assertIn("r['invalid_at']", window)
        self.assertIn("IS NOT NULL", window)
        self.assertNotIn("$since", window)
        self.assertNotIn("$until", window)
        self.assertNotIn(">=", window)
        with self.assertRaises(InvalidCypherIdentifier):
            relationship_valid_at_clause("r`", "as_of")
        with self.assertRaises(InvalidCypherIdentifier):
            relationship_changed_in_window_clause("r", "since; MATCH", "until")

    def test_bound_integers(self):
        self.assertEqual(require_limit(10), 10)
        self.assertEqual(require_int("3", name="depth", maximum=10), 3)
        with self.assertRaises(InvalidCypherIdentifier):
            require_limit(-1)
        with self.assertRaises(InvalidCypherIdentifier):
            require_limit(True)
        with self.assertRaises(InvalidCypherIdentifier):
            require_int("1; MATCH", name="limit")
        with self.assertRaises(InvalidCypherIdentifier):
            require_limit("²")
        with self.assertRaises(InvalidCypherIdentifier):
            require_limit("1" * 10_000)

    def test_validate_raw_cypher_allows_reads_and_rejects_admin(self):
        self.assertEqual(validate_raw_cypher("RETURN 1;"), "RETURN 1")
        self.assertEqual(
            validate_raw_cypher("MATCH (n) RETURN n LIMIT 5"),
            "MATCH (n) RETURN n LIMIT 5",
        )
        self.assertEqual(
            validate_raw_cypher("RETURN 'USE system' AS note"),
            "RETURN 'USE system' AS note",
        )
        self.assertIn("db.labels", validate_raw_cypher("CALL db.labels() YIELD label RETURN label"))
        rejected = [
            "USE system MATCH (n) RETURN n",
            "CREATE DATABASE other",
            "CREATE OR REPLACE DATABASE other",
            "CREATE OR REPLACE COMPOSITE DATABASE other",
            "DROP DATABASE neo4j",
            "SHOW DATABASES",
            "CALL dbms.components() YIELD name RETURN name",
            "CALL db.admin.waitForIndexes() YIELD name RETURN name",
            "CALL `dbms`.components() YIELD name RETURN name",
            "CALL `db`.`admin`.waitForIndexes() YIELD name RETURN name",
            r"WITH 'a\'' AS s CALL dbms.components() YIELD name RETURN name",
            "WITH '' AS s CALL dbms.components() YIELD name RETURN name",
            'WITH "" AS s CALL dbms.components() YIELD name RETURN name',
            "WITH '' AS s USE system RETURN 1",
            "CALL dbms/*x*/.components() YIELD name RETURN name",
            "CALL dbms\n.components() YIELD name RETURN name",
            "CALL dbms .components() YIELD name RETURN name",
            "CALL `dbms`/*x*/.components() YIELD name RETURN name",
            "CALL db./*x*/admin.waitForIndexes() YIELD name RETURN name",
            "CALL db.\nadmin.waitForIndexes() YIELD name RETURN name",
            "RETURN 1; MATCH (n) DETACH DELETE n",
            "",
        ]
        for query in rejected:
            with self.subTest(query=query):
                with self.assertRaises(RawCypherRejected):
                    validate_raw_cypher(query)

    def test_validate_raw_cypher_allows_empty_strings_and_db_labels(self):
        self.assertEqual(validate_raw_cypher("RETURN '' AS s"), "RETURN '' AS s")
        self.assertEqual(
            validate_raw_cypher("WITH '' AS s RETURN s"),
            "WITH '' AS s RETURN s",
        )
        self.assertIn(
            "db.labels",
            validate_raw_cypher("CALL db.labels() YIELD label RETURN label"),
        )

    def test_validate_raw_cypher_ignores_quoted_labels_for_admin_keywords(self):
        self.assertEqual(
            validate_raw_cypher("MATCH (n:`USE`) RETURN n"),
            "MATCH (n:`USE`) RETURN n",
        )
        self.assertIn(
            "db.labels",
            validate_raw_cypher("CALL db.labels() YIELD label RETURN label"),
        )


if __name__ == "__main__":
    unittest.main()
