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


def test_worker_commits_only_while_claim_is_current(monkeypatch):
    from src.core.saving import ingest_worker

    monkeypatch.setattr(ingest_worker.ingest_staging, "claim_unit", lambda *_: "attempt")
    renew = Mock(side_effect=[False, True])
    monkeypatch.setattr(ingest_worker.ingest_staging, "renew_claim", renew)
    commit = Mock(return_value=True)
    monkeypatch.setattr(ingest_worker, "commit_output", commit)
    process = Mock()

    assert not ingest_worker.execute_staged_unit("brain", "root", "unit", process)
    commit.assert_not_called()
    assert ingest_worker.execute_staged_unit("brain", "root", "unit", process)
    commit.assert_called_once_with("brain", "root", "unit", "attempt", [])


def test_run_publisher_holds_gate_through_completion(monkeypatch):
    from contextlib import contextmanager
    from src.core.saving import ingest_publication as publisher

    events = []

    @contextmanager
    def gate(brain, root):
        events.append(("enter", brain, root))
        yield
        events.append(("exit", brain, root))

    def publish_unit(brain, root, replay):
        if events[-1][0] == "replay":
            return False
        replay(brain, "op:0", {"adapter": "data"})
        return True

    monkeypatch.setattr(publisher, "publish_next_unit", publish_unit)
    monkeypatch.setattr(
        publisher, "replay_data_operation",
        lambda client, brain, operation: events.append(("replay", brain, operation)),
    )
    monkeypatch.setattr(
        publisher, "finish_publication",
        lambda brain, root: events.append(("finish", brain, root)),
    )

    monkeypatch.setattr(
        publisher, "discard_outputs",
        lambda brain, root: events.append(("cleanup", brain, root)),
    )
    publisher.publish_staged_run("brain", "root", gate, Mock(), Mock())
    assert [event[0] for event in events] == ["enter", "replay", "cleanup", "finish", "exit"]


def test_http_gate_spans_response_and_allows_task_polling(monkeypatch):
    import asyncio
    from contextlib import contextmanager
    from src.services.api.middlewares import ingest_publication as middleware

    events = []

    @contextmanager
    def gate(brain):
        events.append(("enter", brain))
        yield
        events.append(("exit", brain))

    async def app(scope, receive, send):
        events.append(("response", scope["path"]))

    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setattr(middleware, "brain_access", gate)
    guarded = middleware.IngestPublicationMiddleware(app)
    scope = {"type": "http", "method": "GET", "state": {"brain_id": "brain"}}
    asyncio.run(guarded({**scope, "path": "/retrieve"}, None, None))
    asyncio.run(guarded({**scope, "path": "/tasks/task"}, None, None))
    assert events == [
        ("enter", "brain"), ("response", "/retrieve"), ("exit", "brain"),
        ("response", "/tasks/task"),
    ]


def test_staged_index_and_vector_removal_do_not_touch_live_stores():
    from src.adapters.embeddings import VectorStoreAdapter
    from src.adapters.graph import GraphAdapter
    from src.core.saving.ingest_publication import replay_graph_operation, replay_vector_operation
    from src.core.saving.ingest_write_buffer import capture_writes

    graph = GraphAdapter()
    graph.graph = Mock()
    vectors = VectorStoreAdapter()
    vectors.add_client(Mock())

    with capture_writes("brain") as operations:
        graph.refresh_hub_bridges_for_entities(["node"], "brain")
        vectors.remove_vectors(["vector"], "data", "brain")

    graph.graph.refresh_hub_bridges_for_entities.assert_not_called()
    vectors.vector_store.remove_vectors.assert_not_called()
    replay_graph_operation(graph.graph, "brain", operations[0])
    replay_vector_operation(vectors.vector_store, "brain", operations[1])
    graph.graph.refresh_hub_bridges_for_entities.assert_called_once_with("brain", ["node"])
    vectors.vector_store.remove_vectors.assert_called_once_with(["vector"], "data", "brain")


def test_staging_context_can_capture_executor_writes():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from src.core.saving.ingest_write_buffer import capture_operation, capture_writes

    operation = {"adapter": "vector", "method": "remove_vectors"}
    with capture_writes("brain") as operations, ThreadPoolExecutor() as executor:
        future = executor.submit(copy_context().run, capture_operation, "brain", operation)
        assert future.result()
    assert operations == [operation]


def test_staging_vector_lookup_sees_inserts_and_removals():
    from src.adapters.embeddings import VectorStoreAdapter
    from src.constants.embeddings import Vector
    from src.core.saving.ingest_write_buffer import capture_writes

    adapter = VectorStoreAdapter()
    adapter.add_client(Mock(get_by_ids=Mock(return_value=[])))
    vector = Vector(id="stable", embeddings=[1.0], metadata={})
    with capture_writes("brain"):
        ids = adapter.add_vectors([vector], "relationships", "brain")
        assert adapter.get_by_ids(ids, "relationships", "brain") == [vector]
        adapter.remove_vectors(ids, "relationships", "brain")
        assert adapter.get_by_ids(ids, "relationships", "brain") == []
    adapter.vector_store.get_by_ids.assert_not_called()
    adapter.get_by_ids(ids, "relationships", "brain")
    adapter.vector_store.get_by_ids.assert_called_once_with(ids, "relationships", "brain")
