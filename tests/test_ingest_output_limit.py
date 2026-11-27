import os
from unittest.mock import MagicMock, patch

import pytest

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

from src.core.agents.architect_agent import ArchitectOutputLimitError
from src.core.agents.scout_agent import ScoutEntity
from src.core.saving import auto_kg
from src.core.saving.ingest_limits import bind_output_cap, limit_output


class _CapModel:
    model_fields = {
        "max_tokens": object(),
        "max_output_tokens": object(),
        "model_kwargs": object(),
    }

    def __init__(self, max_tokens=16, max_output_tokens=16, model_kwargs=None):
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.model_kwargs = dict(model_kwargs or {})

    def model_copy(self, update=None):
        data = {
            "max_tokens": self.max_tokens,
            "max_output_tokens": self.max_output_tokens,
            "model_kwargs": dict(self.model_kwargs),
        }
        data.update(update or {})
        return _CapModel(**data)


def _tiny_text():
    return "Tiny file body. " * 12


def _enrich_with_architect(architect, text):
    scout = MagicMock()
    scout._run_chunk.return_value.entities = [
        ScoutEntity(name="Tiny", type="Document"),
    ]
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
        patch.object(auto_kg.config, "ingest_architect_dense_entity_threshold", 99),
        patch.object(auto_kg.config, "ingest_max_new_tokens", 8192),
    ):
        return auto_kg.enrich_kg_from_input(text, brain_id="brain-tiny")


def test_output_cap_is_applied_when_bound():
    model = _CapModel()
    with bind_output_cap(8192):
        capped = limit_output(model)
    assert capped is not model
    assert capped.max_tokens == 8192
    assert capped.max_output_tokens == 8192
    assert capped.model_kwargs["max_tokens"] == 8192


def test_output_cap_zero_leaves_model_unchanged():
    model = _CapModel()
    with bind_output_cap(0):
        assert limit_output(model) is model


def test_architect_output_limit_error_has_stable_code():
    error = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    assert error.code == "extraction_output_limit"


def test_tiny_file_keeps_already_extracted_at_split_floor():
    text = _tiny_text()
    assert len(text.strip()) <= 256
    architect = MagicMock()
    architect.relationships_set = [MagicMock(name="kept")]
    architect.pending_persistence_batches = []
    architect.run_batch_extract.side_effect = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    architect.run_tooler.side_effect = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    _enrich_with_architect(architect, text)
    architect.run_tooler.assert_called_once()


def test_tiny_file_falls_back_to_tooler_at_split_floor():
    text = _tiny_text()
    architect = MagicMock()
    architect.relationships_set = []
    architect.pending_persistence_batches = []
    architect.run_batch_extract.side_effect = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    _enrich_with_architect(architect, text)
    architect.run_tooler.assert_called_once()
    assert architect.run_tooler.call_args.kwargs["reset"] is True


def test_unrecoverable_tiny_file_raises_extraction_output_limit():
    text = _tiny_text()
    architect = MagicMock()
    architect.relationships_set = []
    architect.pending_persistence_batches = []
    architect.run_batch_extract.side_effect = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    architect.run_tooler.side_effect = ArchitectOutputLimitError(
        "Relationship extraction exceeded the model output limit."
    )
    with pytest.raises(ArchitectOutputLimitError) as raised:
        _enrich_with_architect(architect, text)
    assert raised.value.code == "extraction_output_limit"