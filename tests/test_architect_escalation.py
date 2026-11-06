import os

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "PIPELINE_MODE": "accurate",
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

from unittest.mock import MagicMock, patch

from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.scout_agent import ScoutEntity
from src.core.saving.architect_batch import BatchExtractResponse
from src.core.saving.auto_kg import _dense_architect_parts, _filter_entities_for_text
from src.core.saving.ingest_cost import IngestCostLedger
from src.utils.text_chunking import chunk_text


def _entity(uuid: str, name: str, type_: str = "Event"):
    return ScoutEntity(uuid=uuid, name=name, type=type_)


def test_chunk_text_splits_dialogue_on_single_newlines():
    lines = [f"Speaker: turn number {i} with some filler text." for i in range(20)]
    text = "\n".join(lines)
    parts = chunk_text(text, max_chars=200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    joined = "\n".join(parts)
    assert "Speaker: turn number 0" in joined
    assert "Speaker: turn number 19" in joined


def test_dense_architect_parts_splits_entity_heavy_dialogue():
    lines = [
        "Caroline: I applied to adoption agencies this week.",
        "Melanie: That is huge. How are Oscar and Bailey?",
        "Caroline: Oscar hid a bone. Bailey is new.",
        "Melanie: Show me Oliver too.",
        "Caroline: I also painted a horse with dad.",
        "Melanie: Proud of you.",
    ]
    text = "\n".join(lines)
    entities = [
        _entity(f"e{i}", name)
        for i, name in enumerate(
            [
                "Caroline",
                "Melanie",
                "Applied to adoption agencies",
                "Oscar",
                "Bailey",
                "Oliver",
                "Hid a bone",
                "Painted a horse",
                "Horseback riding",
                "Self-portrait",
                "Goodbyes",
                "Support",
                "Conversation",
            ]
        )
    ]
    assert len(entities) >= 12
    parts = _dense_architect_parts(
        text, entities, entity_threshold=12, dense_max_chars=120
    )
    assert len(parts) > 1
    filtered = _filter_entities_for_text(entities, parts[0])
    assert filtered
    assert len(filtered) <= len(entities)


def test_run_batch_extract_empty_with_escalate_disabled_stays_partial():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice met Bob."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Bob", type="PERSON"),
    ]
    empty = BatchExtractResponse(relationships=[], new_nodes=[])

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler"
    ) as tooler:
        agent.agent = MagicMock()
        agent.agent.invoke.return_value = {
            "structured_response": empty,
            "messages": [],
        }
        ledger = IngestCostLedger()
        rels = agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=False,
            timeout=5,
            max_retries=1,
        )
        tooler.assert_not_called()
        assert rels == []
        assert ledger.architect_escalations == 0


def test_run_batch_extract_escalate_respects_zero_turn_budget():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice met Bob."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Bob", type="PERSON"),
    ]
    empty = BatchExtractResponse(relationships=[], new_nodes=[])

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler"
    ) as tooler, patch(
        "src.core.agents.architect_agent.config"
    ) as mock_config:
        mock_config.ingest_architect_max_schema_calls = 2
        mock_config.ingest_architect_escalate_max_turns = 0
        agent.agent = MagicMock()
        agent.agent.invoke.return_value = {
            "structured_response": empty,
            "messages": [],
        }
        ledger = IngestCostLedger()
        rels = agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=True,
            timeout=5,
            max_retries=1,
        )
        tooler.assert_not_called()
        assert rels == []
        assert ledger.architect_escalations == 1
        assert "escalate_budget_exhausted" in ledger.janitor_drop_reasons


def test_run_batched_janitor_caps_llm_calls():
    from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship

    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )

    pending = [
        ArchitectAgentRelationship(
            flow_key=f"f{i}",
            tail=ArchitectAgentEntity(uuid=f"t{i}", name=f"Tail{i}", type="PERSON"),
            tip=ArchitectAgentEntity(uuid=f"p{i}", name=f"Tip{i}", type="PERSON"),
            name="RELATED",
            description="not in source text at all",
            properties={},
        )
        for i in range(5)
    ]
    agent.pending_persistence_batches = [pending]
    agent.relationships_set = list(pending)

    janitor = MagicMock()
    janitor.run_atomic_janitor.return_value = "OK"
    agent.janitor_agent = janitor
    agent._janitor_agent_brain_id = "default"

    ledger = IngestCostLedger()
    with patch(
        "src.core.agents.architect_agent.config"
    ) as mock_config, patch(
        "src.core.saving.grounding.triage_relationships_for_janitor"
    ) as triage_mock:
        mock_config.ingest_janitor_max_llm_calls = 1
        triage = MagicMock()
        triage.accept = []
        triage.reject = []
        triage.ambiguous = pending
        triage_mock.return_value = triage
        agent.run_batched_janitor(
            text="Alice met Bob.",
            brain_id="default",
            batch_size=2,
            cost_ledger=ledger,
        )

    assert janitor.run_atomic_janitor.call_count == 1
    assert any(
        str(r).startswith("janitor_budget_exhausted")
        for r in ledger.janitor_drop_reasons
    )


def test_truncated_bedrock_batch_is_reported_before_tool_escalation():
    import pytest

    from src.core.agents.architect_agent import ArchitectOutputLimitError

    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = {
        "messages": [{"response_metadata": {"stopReason": "max_tokens"}}]
    }
    with patch.object(agent, "_get_agent"), patch.object(agent, "run_tooler") as tooler:
        with pytest.raises(ArchitectOutputLimitError):
            agent.run_batch_extract("Alice met Bob.", [_entity("1", "Alice", "PERSON")])
        tooler.assert_not_called()
