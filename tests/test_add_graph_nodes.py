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

from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
from src.constants.kg import Node
from src.workers.tasks.ingestion import (
    add_graph_nodes,
    graph_ingest_account,
    labels_for_graph_node,
)


class _Adapter:
    def __init__(self):
        self.added = []

    def add_nodes(self, nodes, brain_id=None):
        self.added = list(nodes)
        return list(nodes)


def _relationship(*, tail, tip, uuid="e-1", name="WORKS_ON"):
    return ArchitectAgentRelationship(
        uuid=uuid,
        flow_key="flow-1",
        name=name,
        tail=tail,
        tip=tip,
        properties={"v_id": "v-rel-1"},
    )


def _fill_written_edge(ledger, relationship, *, created=True):
    if ledger is None:
        return None
    ledger["edges"] = [
        {
            "uuid": relationship.uuid,
            "created": created,
            "v_id": (relationship.properties or {}).get("v_id"),
            "name": relationship.name,
        }
    ]
    return ledger


class AddGraphNodesLabelTests(unittest.TestCase):
    def test_mixed_valid_and_invalid_extra_labels_keep_the_node(self):
        self.assertEqual(
            labels_for_graph_node(["PERSON", "CI/CD", "ENGINEER", "R&D"]),
            ["PERSON", "ENGINEER"],
        )
        adapter = _Adapter()
        node = Node(
            uuid="n-pipeline",
            name="CI Pipeline",
            labels=["PERSON", "CI/CD", "ENGINEER", "R&D"],
            properties={"v_id": "v-n1", "catalog_labels": ["CI/CD", "ENGINEER", "R&D"]},
        )
        added = add_graph_nodes(adapter, [node], "brain")
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].uuid, "n-pipeline")
        self.assertEqual(added[0].labels, ["PERSON", "ENGINEER"])
        self.assertEqual(adapter.added[0].labels, ["PERSON", "ENGINEER"])
        self.assertEqual(adapter.added[0].name, "CI Pipeline")

    def test_invalid_primary_label_skips_the_node(self):
        adapter = _Adapter()
        node = Node(
            uuid="n-rnd",
            name="R&D",
            labels=["R&D"],
            properties={"v_id": "v-rnd"},
        )
        added = add_graph_nodes(adapter, [node], "brain")
        self.assertEqual(added, [])
        self.assertEqual(adapter.added, [])


class GraphIngestLedgerTests(unittest.TestCase):
    def test_skipped_endpoint_is_an_item_error_and_omits_ledger_edge(self):
        adapter = _Adapter()
        skipped = Node(uuid="n-rnd", name="R&D", labels=["R&D"], properties={"v_id": "v-rnd"})
        added = add_graph_nodes(adapter, [skipped], "brain")
        relationship = _relationship(
            tail=ArchitectAgentEntity(
                uuid="n-rnd",
                name="R&D",
                type="R&D",
                properties={"v_id": "v-rnd"},
            ),
            tip=ArchitectAgentEntity(
                uuid="n-alice",
                name="Alice",
                type="PERSON",
                properties={"v_id": "v-alice"},
            ),
        )
        errors, ledger, write_edge = graph_ingest_account(
            relationship,
            requested_new_nodes=[skipped],
            added_nodes=added,
            subject_exists=False,
            object_exists=True,
        )
        self.assertFalse(write_edge)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["uuid"], "n-rnd")
        self.assertEqual(errors[0]["node"], "R&D")
        self.assertIn("invalid graph label", errors[0]["message"])
        self.assertIsNone(ledger)

    def test_ledger_matches_nodes_and_edges_actually_written(self):
        adapter = _Adapter()
        mixed = Node(
            uuid="n-pipeline",
            name="CI Pipeline",
            labels=["PERSON", "CI/CD"],
            properties={"v_id": "v-n1"},
        )
        added = add_graph_nodes(adapter, [mixed], "brain")
        relationship = _relationship(
            tail=ArchitectAgentEntity(
                uuid="n-pipeline",
                name="CI Pipeline",
                type="PERSON",
                properties={"v_id": "v-n1"},
            ),
            tip=ArchitectAgentEntity(
                uuid="n-alice",
                name="Alice",
                type="PERSON",
                properties={"v_id": "v-alice"},
            ),
        )
        errors, ledger, write_edge = graph_ingest_account(
            relationship,
            requested_new_nodes=[mixed],
            added_nodes=added,
            subject_exists=False,
            object_exists=True,
        )
        self.assertEqual(errors, [])
        self.assertTrue(write_edge)
        written = _fill_written_edge(ledger, relationship, created=True)
        self.assertEqual(
            [touch["uuid"] for touch in written["nodes"]],
            ["n-pipeline", "n-alice"],
        )
        self.assertEqual(
            [touch["created"] for touch in written["nodes"]],
            [True, False],
        )
        self.assertEqual(written["edges"][0]["uuid"], relationship.uuid)
        self.assertTrue(written["edges"][0]["created"])
        self.assertEqual({node.uuid for node in adapter.added}, {"n-pipeline"})
        self.assertEqual(adapter.added[0].labels, ["PERSON"])
        written_ids = {touch["uuid"] for touch in written["nodes"]} | {
            edge["uuid"] for edge in written["edges"]
        }
        self.assertEqual(written_ids, {"n-pipeline", "n-alice", relationship.uuid})

    def test_written_node_is_ledgered_when_the_other_endpoint_is_skipped(self):
        adapter = _Adapter()
        kept = Node(
            uuid="n-alice",
            name="Alice",
            labels=["PERSON", "CI/CD"],
            properties={"v_id": "v-alice"},
        )
        skipped = Node(
            uuid="n-rnd",
            name="R&D",
            labels=["R&D"],
            properties={"v_id": "v-rnd"},
        )
        added = add_graph_nodes(adapter, [kept, skipped], "brain")
        relationship = _relationship(
            tail=ArchitectAgentEntity(
                uuid="n-alice",
                name="Alice",
                type="PERSON",
                properties={"v_id": "v-alice"},
            ),
            tip=ArchitectAgentEntity(
                uuid="n-rnd",
                name="R&D",
                type="R&D",
                properties={"v_id": "v-rnd"},
            ),
        )
        errors, ledger, write_edge = graph_ingest_account(
            relationship,
            requested_new_nodes=[kept, skipped],
            added_nodes=added,
            subject_exists=False,
            object_exists=False,
        )
        self.assertFalse(write_edge)
        self.assertEqual([error["uuid"] for error in errors], ["n-rnd"])
        self.assertEqual([touch["uuid"] for touch in ledger["nodes"]], ["n-alice"])
        self.assertEqual(ledger["edges"], [])
        self.assertEqual([node.uuid for node in adapter.added], ["n-alice"])
        self.assertEqual(adapter.added[0].labels, ["PERSON"])


if __name__ == "__main__":
    unittest.main()
