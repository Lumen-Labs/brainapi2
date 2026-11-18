import os
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
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


def _triple(happened_at="01/01/2024", subject_uuid=None, event_uuid=None):
    subject = {"name": "Alice", "type": "PERSON"}
    if subject_uuid:
        subject["uuid"] = subject_uuid
    event = {
        "name": "Purchase",
        "type": "EVENT",
        "happened_at": happened_at,
    }
    if event_uuid:
        event["uuid"] = event_uuid
    return {
        "subject": subject,
        "subj_event": {"name": "MADE"},
        "event": event,
        "event_obj": {"name": "TARGETED"},
        "object": {"name": "Widget", "type": "PRODUCT"},
    }


class StableIdentityHelpersTests(unittest.TestCase):
    def test_stable_node_id_is_deterministic(self):
        from src.core.saving.identity import stable_node_id

        first = stable_node_id("Alice", "PERSON")
        second = stable_node_id("alice", "person")
        self.assertEqual(first, second)
        self.assertNotEqual(
            stable_node_id("Purchase", "EVENT", "01/01/2024"),
            stable_node_id("Purchase", "EVENT", "02/01/2024"),
        )

    def test_supplied_uuid_is_authoritative(self):
        from src.core.saving.identity import stable_node_id

        self.assertEqual(
            stable_node_id("Alice", "PERSON", supplied_uuid="keep-me"),
            "keep-me",
        )


class TripleReplayIdentityTests(unittest.TestCase):
    def test_identical_structured_conversion_replays_same_ids(self):
        from src.core.agents.architect_agent import ingestion_triples_to_relationships
        from src.services.api.constants.requests import IngestionTripleSet

        triple = IngestionTripleSet(**_triple())
        first_rels, first_entities = ingestion_triples_to_relationships([triple], [])
        second_rels, second_entities = ingestion_triples_to_relationships([triple], [])

        self.assertEqual(
            [r.uuid for r in first_rels],
            [r.uuid for r in second_rels],
        )
        self.assertEqual(
            {k: v.uuid for k, v in first_entities.items()},
            {k: v.uuid for k, v in second_entities.items()},
        )
        self.assertEqual(first_rels[0].tip.uuid, first_rels[1].tail.uuid)

    def test_same_named_events_with_different_dates_stay_separate(self):
        from src.core.agents.architect_agent import ingestion_triples_to_relationships
        from src.services.api.constants.requests import IngestionTripleSet

        first = IngestionTripleSet(**_triple(happened_at="01/01/2024"))
        second = IngestionTripleSet(**_triple(happened_at="02/01/2024"))
        relationships, entities = ingestion_triples_to_relationships(
            [first, second], []
        )

        event_uuids = {
            rel.tip.uuid if rel.name == "MADE" else rel.tail.uuid
            for rel in relationships
            if rel.name in ("MADE", "TARGETED")
        }
        purchase_events = [
            entity
            for key, entity in entities.items()
            if len(key) >= 2 and key[0] == "purchase" and key[1] == "event"
        ]
        self.assertEqual(len(purchase_events), 2)
        self.assertEqual(len({e.uuid for e in purchase_events}), 2)
        self.assertEqual(len(event_uuids), 2)


class ResolutionIdentityTests(unittest.TestCase):
    def test_existing_uuid_is_reused_and_not_overwritten(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.constants.kg import Node
        from src.workers.tasks import ingestion as ingestion_mod

        existing = Node(
            uuid="existing-alice",
            name="Alice",
            labels=["PERSON"],
            properties={},
        )
        alice = ArchitectAgentEntity(
            uuid="existing-alice",
            name="Alice",
            type="PERSON",
        )
        widget = ArchitectAgentEntity(
            uuid="widget-1",
            name="Widget",
            type="PRODUCT",
        )
        relationships = [
            ArchitectAgentRelationship(
                tail=alice,
                name="OWNS",
                tip=widget,
                flow_key="flow-1",
                uuid="rel-supplied",
            )
        ]

        with patch.object(
            ingestion_mod.graph_adapter,
            "get_by_uuid",
            side_effect=lambda uuid, brain_id="default": (
                existing if uuid == "existing-alice" else None
            ),
        ), patch.object(
            ingestion_mod.graph_adapter,
            "get_by_identification_params",
            return_value=None,
        ), patch.object(
            ingestion_mod.vector_store_adapter,
            "search_vectors",
            return_value=[],
        ):
            ingestion_mod._resolve_relationship_entities(relationships, "brain-a")

        self.assertEqual(relationships[0].tail.uuid, "existing-alice")
        self.assertEqual(relationships[0].uuid, "rel-supplied")

    def test_ambiguous_semantic_matches_do_not_force_merge(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.constants.embeddings import Vector
        from src.constants.kg import Node
        from src.workers.tasks import ingestion as ingestion_mod

        alice = ArchitectAgentEntity(uuid="new-alice", name="Alice", type="PERSON")
        widget = ArchitectAgentEntity(uuid="widget-1", name="Widget", type="PRODUCT")
        relationships = [
            ArchitectAgentRelationship(
                tail=alice,
                name="OWNS",
                tip=widget,
                flow_key="flow-1",
            )
        ]

        candidate_a = MagicMock()
        candidate_a.id = "va"
        candidate_a.metadata = {"uuid": "a1", "labels": ["PERSON"]}
        candidate_b = MagicMock()
        candidate_b.id = "vb"
        candidate_b.metadata = {"uuid": "a2", "labels": ["PERSON"]}

        with patch.object(
            ingestion_mod.graph_adapter, "get_by_uuid", return_value=None
        ), patch.object(
            ingestion_mod.graph_adapter,
            "get_by_identification_params",
            return_value=None,
        ), patch.object(
            ingestion_mod.embeddings_adapter,
            "embed_text",
            return_value=Vector(id="q", embeddings=[1.0, 0.0, 0.0], metadata={}),
        ), patch.object(
            ingestion_mod.vector_store_adapter,
            "search_vectors",
            return_value=[candidate_a, candidate_b],
        ), patch.object(
            ingestion_mod.vector_store_adapter,
            "get_by_ids",
            side_effect=lambda ids, store="nodes", brain_id="default": [
                Vector(id=ids[0], embeddings=[1.0, 0.0, 0.0], metadata={})
            ],
        ), patch.object(
            ingestion_mod,
            "cosine_similarity",
            return_value=0.95,
        ):
            ingestion_mod._resolve_relationship_entities(relationships, "brain-a")

        self.assertEqual(relationships[0].tail.uuid, "new-alice")

    def test_unparseable_source_timestamp_is_not_copied_to_valid_at(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.workers.tasks import ingestion as ingestion_mod

        relationships = [
            ArchitectAgentRelationship(
                tail=ArchitectAgentEntity(uuid="alice", name="Alice", type="PERSON"),
                name="LIVES_IN",
                tip=ArchitectAgentEntity(uuid="paris", name="Paris", type="CITY"),
                flow_key="flow-1",
            )
        ]
        with patch.object(
            ingestion_mod.graph_adapter, "get_by_uuid", return_value=None
        ), patch.object(
            ingestion_mod.graph_adapter,
            "get_by_identification_params",
            return_value=None,
        ), patch.object(
            ingestion_mod.embeddings_adapter,
            "embed_text",
            side_effect=RuntimeError("no embed"),
        ):
            ingestion_mod._resolve_relationship_entities(
                relationships,
                "brain-a",
                source_timestamp="0000-00-00",
            )
        props = relationships[0].properties or {}
        self.assertEqual(props.get("source_timestamp"), "0000-00-00")
        self.assertNotIn("valid_at", props)
        self.assertNotIn("valid_from", props)

    def test_parseable_source_timestamp_is_copied_to_valid_at(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.workers.tasks import ingestion as ingestion_mod

        relationships = [
            ArchitectAgentRelationship(
                tail=ArchitectAgentEntity(uuid="alice", name="Alice", type="PERSON"),
                name="LIVES_IN",
                tip=ArchitectAgentEntity(uuid="paris", name="Paris", type="CITY"),
                flow_key="flow-1",
                properties={"valid_at": "2023-06-01T00:00:00Z"},
            )
        ]
        stamp = "2024-01-01T00:00:00Z"
        with patch.object(
            ingestion_mod.graph_adapter, "get_by_uuid", return_value=None
        ), patch.object(
            ingestion_mod.graph_adapter,
            "get_by_identification_params",
            return_value=None,
        ), patch.object(
            ingestion_mod.embeddings_adapter,
            "embed_text",
            side_effect=RuntimeError("no embed"),
        ):
            ingestion_mod._resolve_relationship_entities(
                relationships,
                "brain-a",
                source_timestamp=stamp,
            )
            blank = ArchitectAgentRelationship(
                tail=ArchitectAgentEntity(uuid="alice", name="Alice", type="PERSON"),
                name="LIVES_IN",
                tip=ArchitectAgentEntity(uuid="paris", name="Paris", type="CITY"),
                flow_key="flow-1",
            )
            ingestion_mod._resolve_relationship_entities(
                [blank],
                "brain-a",
                source_timestamp=stamp,
            )
        self.assertEqual(
            (relationships[0].properties or {}).get("valid_at"),
            "2023-06-01T00:00:00Z",
        )
        self.assertEqual((blank.properties or {}).get("valid_at"), stamp)
        self.assertEqual((blank.properties or {}).get("valid_from"), stamp)


if __name__ == "__main__":
    unittest.main()
