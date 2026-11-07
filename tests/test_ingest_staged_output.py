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


def test_publication_replays_same_operation_ids_after_failure(monkeypatch):
    from src.core.saving import ingest_publication as publisher

    unit = {"output_ref": "output", "operation_id": "unit", "cursor": 3}
    monkeypatch.setattr(publisher, "next_publication_unit", lambda *_: unit)
    monkeypatch.setattr(publisher, "load_output", lambda *_: ["first", "second"])
    advance = Mock()
    monkeypatch.setattr(publisher, "advance_publication", advance)
    replay = Mock(side_effect=[None, RuntimeError("interrupted"), None, None])

    with pytest.raises(RuntimeError, match="interrupted"):
        publisher.publish_next_unit("brain", "root", replay)
    advance.assert_not_called()
    assert publisher.publish_next_unit("brain", "root", replay)
    assert replay.call_args_list[:2] == replay.call_args_list[2:]
    advance.assert_called_once_with("brain", "root", 3)


def test_staged_vector_write_uses_replayable_identity():
    from src.adapters.embeddings import VectorStoreAdapter
    from src.constants.embeddings import Vector
    from src.core.saving.ingest_publication import replay_vector_operation
    from src.core.saving.ingest_write_buffer import capture_writes
    from src.lib.postgresql.vectors import string_to_int64

    adapter = VectorStoreAdapter()
    client = Mock()
    adapter.add_client(client)
    vector = Vector(id="stable", embeddings=[1.0], metadata={})
    with capture_writes("brain") as operations:
        assert adapter.add_vectors([vector], "data", "brain") == [string_to_int64("stable")]
    client.add_vectors.assert_not_called()

    replay_vector_operation(client, "brain", operations[0])
    replay_vector_operation(client, "brain", operations[0])
    assert client.upsert_staged_vectors.call_count == 2
    client.upsert_staged_vectors.assert_called_with([vector], "data", "brain")
