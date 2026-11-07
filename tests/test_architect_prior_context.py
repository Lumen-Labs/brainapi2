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

import pytest

from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
from src.core.agents.scout_agent import ScoutEntity
from src.core.saving.architect_scratchpad import (
    build_scratchpad,
    fetch_prior_span,
    format_unit_with_scratchpad,
    resolve_prior_context_mode,
    serialize_scratchpad,
)
from src.core.saving.auto_kg import (
    _architect_unit_text,
    _architect_unit_text_scratchpad,
)
from src.core.saving.ingest_cost import count_source_tokens


def test_resolve_prior_context_defaults_scratchpad_for_batch():
    assert resolve_prior_context_mode("auto", "batch") == "scratchpad"
    assert resolve_prior_context_mode("auto", "schema") == "scratchpad"
    assert resolve_prior_context_mode("auto", "tooler") == "raw"
    assert resolve_prior_context_mode("raw", "batch") == "raw"
    assert resolve_prior_context_mode("scratchpad", "tooler") == "scratchpad"


def test_build_and_serialize_scratchpad_includes_hubs_and_predicates():
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(
            uuid="u2",
            name="Hiring",
            type="EVENT",
            happened_at="2024-01-15",
            description="Alice hired Bob",
        ),
        ScoutEntity(uuid="u3", name="Bob", type="PERSON"),
    ]
    rels = [
        ArchitectAgentRelationship(
            uuid="r1",
            flow_key="fk1",
            name="MADE",
            tail=ArchitectAgentEntity(uuid="u1", name="Alice", type="PERSON"),
            tip=ArchitectAgentEntity(
                uuid="u2",
                name="Hiring",
                type="EVENT",
                happened_at="2024-01-15",
                description="Alice hired Bob",
            ),
            properties={"source_span": "Alice hired Bob as a contractor yesterday"},
        ),
        ArchitectAgentRelationship(
            uuid="r2",
            flow_key="fk2",
            name="TARGETED",
            tail=ArchitectAgentEntity(
                uuid="u2", name="Hiring", type="EVENT", happened_at="2024-01-15"
            ),
            tip=ArchitectAgentEntity(uuid="u3", name="Bob", type="PERSON"),
            properties={"source_span": "hired Bob as a contractor"},
        ),
    ]

    pad = build_scratchpad(entities, rels)
    assert any(e.name == "Alice" for e in pad.entities)
    assert any(h.name == "Hiring" for h in pad.event_hubs)
    hub = next(h for h in pad.event_hubs if h.name == "Hiring")
    assert "Alice" in hub.actors
    assert "Bob" in hub.objects
    assert "2024-01-15" in pad.dates
    assert any(p.predicate == "MADE" for p in pad.recent_predicates)

    text, tokens = serialize_scratchpad(pad, token_cap=500)
    assert tokens > 0
    assert tokens <= 500
    assert "Hiring" in text
    assert "Alice" in text
    assert "MADE" in text
    # Span pointers are truncated quotes, not full prior unit bodies.
    assert "Alice hired Bob" in text


def test_serialize_scratchpad_obeys_token_cap():
    entities = [
        ScoutEntity(
            uuid=f"u{i}",
            name=f"PersonName{i}WithExtraDetail",
            type="PERSON",
            happened_at=f"2024-01-{(i % 28) + 1:02d}",
        )
        for i in range(80)
    ]
    rels = [
        ArchitectAgentRelationship(
            uuid=f"r{i}",
            flow_key=f"fk{i}",
            name="MADE",
            tail=ArchitectAgentEntity(
                uuid=f"u{i}", name=f"PersonName{i}WithExtraDetail", type="PERSON"
            ),
            tip=ArchitectAgentEntity(
                uuid=f"e{i}",
                name=f"EventHub{i}LongLabel",
                type="EVENT",
                happened_at=f"2024-02-{(i % 28) + 1:02d}",
            ),
            properties={
                "source_span": (
                    f"PersonName{i}WithExtraDetail did something notable "
                    f"during EventHub{i}LongLabel on a long day of narrative text."
                )
            },
        )
        for i in range(40)
    ]
    pad = build_scratchpad(entities, rels)
    text, tokens = serialize_scratchpad(pad, token_cap=120)
    assert tokens <= 120
    counted, _, _ = count_source_tokens(text)
    assert counted <= 120


def test_scratchpad_unit_text_excludes_full_prior_bodies():
    prior_body = (
        "Caroline told Melanie she started a new job at Google last Monday "
        "and also mentioned her sister's wedding plans for June in detail. "
        * 8
    )
    prior_entities = [
        ScoutEntity(uuid="u1", name="Caroline", type="PERSON"),
        ScoutEntity(
            uuid="u2",
            name="Started new job",
            type="EVENT",
            happened_at="2023-05-08",
        ),
    ]
    prior_rels = [
        ArchitectAgentRelationship(
            uuid="r1",
            flow_key="fk1",
            name="MADE",
            tail=ArchitectAgentEntity(uuid="u1", name="Caroline", type="PERSON"),
            tip=ArchitectAgentEntity(
                uuid="u2",
                name="Started new job",
                type="EVENT",
                happened_at="2023-05-08",
            ),
            properties={"source_span": "Caroline told Melanie she started a new job"},
        )
    ]
    current = "She said the salary was better than before."
    scratch = _architect_unit_text_scratchpad(
        current,
        prior_entities=prior_entities,
        prior_relationships=prior_rels,
        token_cap=500,
    )
    raw = _architect_unit_text(current, [prior_body])

    assert "[Prior scratchpad]" in scratch
    assert "[Current unit]" in scratch
    assert current in scratch
    assert "Caroline" in scratch
    assert "Started new job" in scratch
    # Must not replay the full prior narrative body.
    assert prior_body not in scratch
    assert prior_body in raw
    assert "sister's wedding plans for June in detail" not in scratch


def test_fetch_prior_span_returns_window():
    prior = [
        "Intro filler. Alice met Bob at the cafe on Tuesday afternoon. More filler."
    ]
    span = fetch_prior_span(prior, "Alice met Bob", window_chars=80)
    assert span is not None
    assert "Alice met Bob" in span
    assert len(span) < len(prior[0])


def test_format_unit_with_scratchpad_empty_passthrough():
    assert format_unit_with_scratchpad("only current", "") == "only current"


def test_enrich_loop_uses_scratchpad_not_raw_prior_for_batch():
    """Integration-ish: second unit prompt must not contain prior chunk body."""
    from src.core.saving import auto_kg

    prior_chunk = (
        "UNIQUE_PRIOR_MARKER_XYZ Caroline started working at Google last week "
        "and talked about her sister wedding plans extensively across many sentences."
    )
    current_chunk = "She said the pay was higher."

    scout = MagicMock()
    first_entities = [
        ScoutEntity(uuid="u1", name="Caroline", type="PERSON"),
        ScoutEntity(uuid="u2", name="Started job", type="EVENT"),
    ]
    second_entities = [
        ScoutEntity(uuid="u1", name="Caroline", type="PERSON"),
    ]

    class _Resp:
        def __init__(self, entities):
            self.entities = entities

    scout._run_chunk.side_effect = [
        _Resp(first_entities),
        _Resp(second_entities),
    ]

    architect = MagicMock()
    architect.session_id = "s1"
    architect.relationships_set = [
        ArchitectAgentRelationship(
            uuid="r1",
            flow_key="fk1",
            name="MADE",
            tail=ArchitectAgentEntity(uuid="u1", name="Caroline", type="PERSON"),
            tip=ArchitectAgentEntity(uuid="u2", name="Started job", type="EVENT"),
            properties={"source_span": "Caroline started working at Google"},
        )
    ]
    architect.pending_persistence_batches = [list(architect.relationships_set)]
    architect.take_pending_relationships.return_value = list(
        architect.relationships_set
    )
    architect.defer_janitor = True

    captured_texts: list[str] = []

    def _capture_batch(text, entities, **kwargs):
        captured_texts.append(text)
        return list(architect.relationships_set)

    architect.run_batch_extract.side_effect = _capture_batch

    with patch.object(auto_kg, "ScoutAgent", return_value=scout), patch.object(
        auto_kg, "ArchitectAgent", return_value=architect
    ), patch.object(auto_kg, "IngestionManager", return_value=MagicMock()), patch.object(
        auto_kg, "source_cut", side_effect=[len(prior_chunk) + 1, len(current_chunk)]
    ), patch.object(
        auto_kg.config, "pipeline_mode", "accurate"
    ), patch.object(
        auto_kg.config, "ingest_architect_mode", "batch"
    ), patch.object(
        auto_kg.config, "ingest_architect_prior_context", "auto"
    ), patch.object(
        auto_kg.config, "ingest_architect_per_unit", True
    ), patch.object(
        auto_kg.config, "ingest_defer_janitor", False
    ), patch.object(
        auto_kg.config, "run_graph_consolidator", False
    ), patch.object(
        auto_kg.config, "ingest_architect_scratchpad_token_cap", 500
    ):
        auto_kg.enrich_kg_from_input(
            prior_chunk + "\n" + current_chunk, brain_id="brain-scratch"
        )

    assert len(captured_texts) == 2
    second = captured_texts[1]
    assert "[Prior scratchpad]" in second
    assert "UNIQUE_PRIOR_MARKER_XYZ" not in second
    assert "wedding plans extensively" not in second
    assert current_chunk in second
    assert "Caroline" in second


def test_enrichment_splits_truncated_relationship_batch_and_preserves_reset():
    from src.core.agents.architect_agent import ArchitectOutputLimitError
    from src.core.saving import auto_kg

    first = "Aurora " * 30 + "Aurora."
    second = "Kepler " * 30 + "Kepler."
    text = first + "\n\n" + second
    scout = MagicMock()
    aurora = ScoutEntity(name="Aurora", type="Project")
    kepler = ScoutEntity(name="Kepler", type="Project")
    scout._run_chunk.side_effect = [
        MagicMock(entities=[aurora, kepler]),
        MagicMock(entities=[aurora]),
        MagicMock(entities=[kepler]),
    ]
    architect = MagicMock()
    architect.run_batch_extract.side_effect = [ArchitectOutputLimitError(), [], []]
    architect.take_pending_relationships.return_value = []
    with (
        patch.object(auto_kg, "ScoutAgent", return_value=scout),
        patch.object(auto_kg, "ArchitectAgent", return_value=architect),
        patch.object(auto_kg, "IngestionManager", return_value=MagicMock()),
        patch.object(auto_kg.config, "pipeline_mode", "accurate"),
        patch.object(auto_kg.config, "ingest_architect_mode", "batch"),
        patch.object(auto_kg.config, "ingest_architect_per_unit", True),
        patch.object(auto_kg.config, "ingest_defer_janitor", False),
        patch.object(auto_kg.config, "run_graph_consolidator", False),
    ):
        auto_kg.enrich_kg_from_input(text, brain_id="brain-retry")

    calls = architect.run_batch_extract.call_args_list
    assert calls[0].args[0] == text
    assert calls[1].args[0].strip() == first
    assert calls[2].args[0].rstrip().endswith(second)
    assert [call.kwargs["reset"] for call in calls] == [True, True, True]
    assert [entity.name for entity in calls[1].args[1]] == ["Aurora"]
    assert [entity.name for entity in calls[2].args[1]] == ["Kepler"]


@pytest.mark.parametrize("dense_scout", [False, True])
def test_model_overflow_retries_in_order_without_collecting_failed_unit(monkeypatch, dense_scout):
    from types import SimpleNamespace

    from src.core.saving import auto_kg
    from src.core.saving.ingest_limits import IngestModelLimitError, invocation_limits, scout_payload_size

    scout = MagicMock()
    architect = MagicMock()
    entity = ScoutEntity(uuid="alice", name="Alice", type="PERSON")
    events = []

    def run_scout(text, **kwargs):
        assert invocation_limits.get() == (128,)
        events.append(("scout", text))
        if dense_scout and len(events) == 1:
            return SimpleNamespace(entities=[
                entity, ScoutEntity(uuid="bob", name="Bob", type="PERSON")
            ])
        return SimpleNamespace(entities=[entity])

    def run_architect(text, entities, **kwargs):
        events.append(("architect", text))
        if dense_scout and scout_payload_size.get()[0] > 1:
            raise IngestModelLimitError("Estimated extraction capacity exceeded")
        if not dense_scout and len(events) == 2:
            raise IngestModelLimitError("Model response exceeded the output limit")

    scout._run_chunk.side_effect = run_scout
    architect.run_batch_extract.side_effect = run_architect
    architect.run_batched_janitor.side_effect = lambda **kw: events.append(
        ("janitor", kw["text"])
    )
    architect.take_pending_relationships.return_value = []
    monkeypatch.setattr(auto_kg, "ScoutAgent", lambda *a, **kw: scout)
    monkeypatch.setattr(auto_kg, "ArchitectAgent", lambda *a, **kw: architect)
    monkeypatch.setattr(auto_kg.config, "pipeline_mode", "accurate")
    monkeypatch.setattr(auto_kg.config, "ingest_architect_mode", "batch")
    monkeypatch.setattr(auto_kg.config, "ingest_architect_per_unit", False)
    monkeypatch.setattr(auto_kg.config, "ingest_defer_janitor", True)

    source = "Alice joined a team.\n\nShe became its lead."
    monkeypatch.setattr(auto_kg.config, "ingest_max_new_tokens", 128)
    auto_kg.enrich_kg_from_input(source)

    assert [stage for stage, _ in events] == [
        "scout", "architect", "scout", "architect", "janitor",
        "scout", "architect", "janitor",
    ]
    assert architect.take_pending_relationships.call_count == 2
    first, second = [text for stage, text in events if stage == "janitor"]
    assert first + second.split("[Current span]\n")[1] == source
    assert "Alice" in second.split("[Current span]\n")[0]
    assert invocation_limits.get() is None
