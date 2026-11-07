from unittest.mock import Mock

import pytest

from src.core.saving import ingest_staged_output as staged


def test_staged_output_is_immutable_and_scoped(monkeypatch):
    documents = {}

    def update_one(query, update, *, upsert):
        assert upsert
        documents.setdefault(query["_id"], {**query, **update["$setOnInsert"]})

    def find_one(query):
        return next(
            (doc for doc in documents.values() if all(doc.get(k) == v for k, v in query.items())),
            None,
        )

    storage = Mock(update_one=update_one, find_one=find_one)
    monkeypatch.setattr(staged, "collection", lambda brain, name: storage)
    operations = [{"adapter": "data", "method": "save_text_chunk", "args": {"id": "one"}}]
    ref = staged.save_output("brain", "root", "unit", "attempt", operations)
    assert staged.save_output("brain", "root", "unit", "attempt", operations) == ref
    assert staged.load_output("brain", "root", ref) == operations
    assert staged.save_output("brain", "root", "unit", "replacement", []) != ref
    with pytest.raises(ValueError):
        staged.save_output("brain", "root", "unit", "attempt", [])
    with pytest.raises(LookupError):
        staged.load_output("brain", "other-root", ref)


def test_data_adapter_captures_writes_in_attempt_scope():
    from src.adapters.data import DataAdapter
    from src.constants.data import TextChunk
    from src.core.saving.ingest_write_buffer import capture_writes

    adapter = DataAdapter()
    adapter.add_client(Mock())
    chunk = TextChunk(text="staged")
    with capture_writes("brain") as operations:
        assert adapter.save_text_chunk(chunk, "brain") is chunk
        assert operations == [{
            "adapter": "data",
            "method": "save_text_chunk",
            "args": {"text_chunk": chunk.model_dump(mode="json")},
        }]
        with pytest.raises(ValueError):
            adapter.save_text_chunk(chunk, "other")
    adapter.data.save_text_chunk.assert_not_called()
    adapter.save_text_chunk(chunk, "brain")
    adapter.data.save_text_chunk.assert_called_once_with(text_chunk=chunk, brain_id="brain")
