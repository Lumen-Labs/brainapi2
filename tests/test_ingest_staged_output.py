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
