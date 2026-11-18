import datetime
import unittest

from src.constants.data import TextChunk
from src.core.saving.source_passages import (
    current_fact_candidates,
    current_record_texts,
    embed_candidate_cap,
    prepare_source_revision,
    prior_revision_texts,
    screen_passage_hits,
    screened_passage_texts,
    source_content_hash,
    visible_passage_body,
)


class _Store:
    def __init__(self, chunks: list[TextChunk] | None = None):
        self.chunks = list(chunks or [])
        self.marked: list[tuple[list[str], str]] = []
        self.limits: list[int] = []

    def get_text_chunks(self, brain_id, limit=10, skip=0, query_text=None, metadata_eq=None, order="desc"):
        self.limits.append(limit)
        matched = []
        for chunk in self.chunks:
            meta = chunk.metadata or {}
            if metadata_eq and any(meta.get(key) != value for key, value in metadata_eq.items()):
                continue
            matched.append(chunk)
        reverse = str(order).lower() != "asc"
        matched.sort(
            key=lambda chunk: getattr(chunk, "inserted_at", None)
            or datetime.datetime.min,
            reverse=reverse,
        )
        return matched[skip : skip + limit], len(matched)

    def mark_text_chunks_superseded(self, ids, superseded_at, brain_id):
        self.marked.append((list(ids), superseded_at))
        for chunk in self.chunks:
            if str(chunk.id) in ids:
                chunk.metadata["superseded_at"] = superseded_at

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        wanted = set(ids)
        return [chunk for chunk in self.chunks if str(chunk.id) in wanted], []


class VisiblePassageTests(unittest.TestCase):
    def test_unmarked_chunk_is_unchanged(self):
        self.assertEqual(
            visible_passage_body({"source": "note"}, "hello", include_history=False),
            "hello",
        )
        self.assertEqual(
            visible_passage_body(None, "hello", include_history=True),
            "hello",
        )

    def test_current_truth_hides_superseded_chunk(self):
        self.assertIsNone(
            visible_passage_body(
                {"superseded_at": "15/11/2026"},
                "old body",
                include_history=False,
            )
        )

    def test_history_marks_superseded_chunk(self):
        self.assertEqual(
            visible_passage_body(
                {"superseded_at": "15/11/2026"},
                "old body",
                include_history=True,
            ),
            "[superseded] old body",
        )


class SourceRevisionTests(unittest.TestCase):
    def test_missing_key_does_not_touch_the_store(self):
        store = _Store()
        meta = {"title": "note"}
        self.assertEqual(
            prepare_source_revision(
                store,
                brain_id="brain",
                source_text="hello",
                meta_keys=meta,
                source_timestamp=None,
            ),
            "plain",
        )
        self.assertNotIn("source_content_hash", meta)
        self.assertEqual(store.marked, [])

    def test_same_hash_does_not_request_a_new_copy(self):
        digest = source_content_hash("hello")
        current = TextChunk(
            id="c1",
            text="hello",
            metadata={"source_key": "src-1", "source_content_hash": digest},
        )
        store = _Store([current])
        meta = {"source_key": "src-1"}
        self.assertEqual(
            prepare_source_revision(
                store,
                brain_id="brain",
                source_text="hello",
                meta_keys=meta,
                source_timestamp="15/11/2026",
            ),
            "unchanged",
        )
        self.assertEqual(store.marked, [])
        self.assertNotIn("superseded_at", current.metadata)

    def test_new_hash_marks_the_previous_chunk_and_keeps_it(self):
        previous = TextChunk(
            id="c1",
            text="hello",
            metadata={"source_key": "src-1", "source_content_hash": "old"},
        )
        store = _Store([previous])
        meta = {"source_key": "src-1"}
        self.assertEqual(
            prepare_source_revision(
                store,
                brain_id="brain",
                source_text="hello again",
                meta_keys=meta,
                source_timestamp="16/11/2026",
            ),
            "insert",
        )
        self.assertEqual(store.marked, [(["c1"], "16/11/2026")])
        self.assertEqual(previous.metadata["superseded_at"], "16/11/2026")
        self.assertEqual(previous.text, "hello")
        self.assertEqual(meta["source_content_hash"], source_content_hash("hello again"))

    def test_unhashed_copies_of_the_same_source_are_all_marked(self):
        copies = [
            TextChunk(id="c1", text="first", metadata={"source_key": "src-1"}),
            TextChunk(id="c2", text="second", metadata={"source_key": "src-1"}),
        ]
        store = _Store(copies)
        meta = {"source_key": "src-1"}
        self.assertEqual(
            prepare_source_revision(
                store,
                brain_id="brain",
                source_text="second",
                meta_keys=meta,
                source_timestamp="16/11/2026",
            ),
            "insert",
        )
        self.assertEqual(len(store.marked), 1)
        self.assertEqual(set(store.marked[0][0]), {"c1", "c2"})
        self.assertEqual(store.marked[0][1], "16/11/2026")

    def test_screen_drops_superseded_hits_unless_history(self):
        live = TextChunk(id="live", text="now", metadata={"source_key": "src-1"})
        old = TextChunk(
            id="old",
            text="then",
            metadata={"source_key": "src-1", "superseded_at": "16/11/2026"},
        )
        store = _Store([live, old])
        hits = [("live", 0.1, "now"), ("old", 0.2, "then"), ("missing", 0.3, "untagged")]
        current = screen_passage_hits(store, "brain", hits, include_history=False)
        self.assertEqual([item[0] for item in current], ["live", "missing"])
        history = screen_passage_hits(store, "brain", hits, include_history=True)
        self.assertEqual(history[1], ("old", 0.2, "[superseded] then"))
        self.assertEqual(history[2][2], "untagged")


class CurrentFactTests(unittest.TestCase):
    def test_drops_facts_whose_chunks_are_all_superseded(self):
        old = TextChunk(
            id="old",
            text="then",
            metadata={"source_key": "src-1", "superseded_at": "16/11/2026"},
        )
        store = _Store([old])
        facts = [
            {"text": "old fact", "chunk_ids": ["old"]},
            {"text": "untagged", "chunk_ids": []},
            {"text": "mixed", "chunk_ids": ["old", "live"]},
        ]
        current = current_fact_candidates(store, "brain", facts, include_history=False)
        self.assertEqual([item["text"] for item in current], ["untagged", "mixed"])
        history = current_fact_candidates(store, "brain", facts, include_history=True)
        self.assertEqual(len(history), 3)

    def test_unstamped_copies_keep_the_newest_chunk_and_its_facts(self):
        older = TextChunk(
            id="old",
            text="The cedar signal code is 4410-OTTER.",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 3, 57),
        )
        middle = TextChunk(
            id="mid",
            text="The cedar signal code is now 7755-PUMA.",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 11, 9),
        )
        newest = TextChunk(
            id="new",
            text="Sync probe cedar was removed.",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 13, 22),
        )
        plain = TextChunk(
            id="plain",
            text="other",
            metadata={},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 0, 0),
        )
        store = _Store([older, middle, newest, plain])
        hits = [
            ("old", 0.1, older.text),
            ("mid", 0.2, middle.text),
            ("new", 0.3, newest.text),
            ("plain", 0.4, plain.text),
        ]
        current = screen_passage_hits(store, "brain", hits, include_history=False)
        self.assertEqual([item[0] for item in current], ["new", "plain"])
        absent = screen_passage_hits(
            store,
            "brain",
            [("old", 0.1, older.text), ("mid", 0.2, middle.text)],
            include_history=False,
        )
        self.assertEqual(absent, [])
        history = screen_passage_hits(store, "brain", hits, include_history=True)
        self.assertEqual([item[0] for item in history], ["old", "mid", "new", "plain"])
        facts = [
            {"text": "puma", "chunk_ids": ["old", "mid"]},
            {"text": "removed", "chunk_ids": ["new"]},
            {"text": "untagged", "chunk_ids": []},
        ]
        kept = current_fact_candidates(store, "brain", facts, include_history=False)
        self.assertEqual([item["text"] for item in kept], ["removed", "untagged"])
        limited = screened_passage_texts(
            store,
            "brain",
            hits,
            include_history=False,
            limit=1,
        )
        self.assertEqual(limited, [newest.text])
        older.text = "Sync probe cedar\n\nThe cedar signal code is 4410-OTTER."
        multiline = screen_passage_hits(
            store,
            "brain",
            [("old", 0.1, older.text), ("new", 0.2, newest.text)],
            include_history=False,
        )
        self.assertEqual([item[0] for item in multiline], ["new"])

    def test_history_loads_a_prior_revision_that_search_missed(self):
        older = TextChunk(
            id="old",
            text="The cedar signal code is 4410-OTTER.",
            metadata={"source_key": "src-1", "superseded_at": "2026-11-16T07:11:00"},
        )
        newest = TextChunk(
            id="new",
            text="Sync probe cedar was removed.",
            metadata={"source_key": "src-1", "operation": "tombstone"},
        )
        store = _Store([older, newest])
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, newest.text)],
            limit=4,
        )
        self.assertEqual(loaded, ["[superseded] The cedar signal code is 4410-OTTER."])

    def test_prior_revisions_rank_by_query_relevance_not_only_recency(self):
        older = TextChunk(
            id="old-code",
            text="The cedar signal code is 4410-OTTER.",
            metadata={"source_key": "src-1", "superseded_at": "2026-11-16T07:11:00"},
        )
        newest_old = TextChunk(
            id="old-snack",
            text="The office snack list was pretzels.",
            metadata={"source_key": "src-1", "superseded_at": "2026-11-16T08:00:00"},
        )
        live = TextChunk(
            id="new",
            text="Sync probe cedar was removed.",
            metadata={"source_key": "src-1"},
        )
        store = _Store([older, newest_old, live])
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
            query_text="what was the cedar signal code",
        )
        self.assertEqual(loaded, ["[superseded] The cedar signal code is 4410-OTTER."])
        by_vector = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
            query_vector=[1.0, 0.0],
            embed=lambda text: [1.0, 0.0] if "4410-OTTER" in text else [0.0, 1.0],
        )
        self.assertEqual(by_vector, ["[superseded] The cedar signal code is 4410-OTTER."])

    def test_prior_revision_embed_calls_are_capped_newest_first(self):
        older = []
        for index in range(40):
            older.append(
                TextChunk(
                    id=f"old-{index}",
                    text=f"revision body {index} unique snack" if index < 34 else f"cedar signal {index}",
                    metadata={
                        "source_key": "src-1",
                        "superseded_at": f"2026-11-16T{index:02d}:00:00",
                    },
                    inserted_at=datetime.datetime(2026, 11, 16, 0, index, 0),
                )
            )
        live = TextChunk(
            id="new",
            text="live copy",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 17, 0, 0, 0),
        )
        store = _Store(older + [live])
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0] if "cedar" in text else [0.0, 1.0]

        limit = 2
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=limit,
            query_vector=[1.0, 0.0],
            embed=embed,
        )
        cap = embed_candidate_cap(limit)
        self.assertEqual(cap, 6)
        self.assertLessEqual(len(calls), cap)
        self.assertGreater(len(calls), 0)
        self.assertEqual(len(loaded), limit)
        self.assertTrue(store.limits)
        self.assertLess(store.limits[-1], 500)
        self.assertLessEqual(store.limits[-1], cap + 16)

    def test_prior_revision_uses_stored_embeddings_without_embed(self):
        older = TextChunk(
            id="old",
            text="The cedar signal code is 4410-OTTER.",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T07:11:00",
                "embeddings": [1.0, 0.0],
            },
        )
        snack = TextChunk(
            id="snack",
            text="The office snack list was pretzels.",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T08:00:00",
                "embeddings": [0.0, 1.0],
            },
        )
        live = TextChunk(id="new", text="live", metadata={"source_key": "src-1"})
        store = _Store([older, snack, live])
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            raise AssertionError("stored embeddings should skip embed_text")

        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
            query_vector=[1.0, 0.0],
            embed=embed,
        )
        self.assertEqual(calls, [])
        self.assertEqual(loaded, ["[superseded] The cedar signal code is 4410-OTTER."])

    def test_prior_revision_batches_embed_texts_once(self):
        older = [
            TextChunk(
                id=f"old-{index}",
                text=f"revision {index} body",
                metadata={
                    "source_key": "src-1",
                    "superseded_at": f"2026-11-16T{index:02d}:00:00",
                },
                inserted_at=datetime.datetime(2026, 11, 16, 0, index, 0),
            )
            for index in range(20)
        ]
        live = TextChunk(
            id="new",
            text="live",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 17, 0, 0, 0),
        )
        store = _Store(older + [live])

        class _Batch:
            def __init__(self):
                self.calls = 0
                self.sizes: list[int] = []

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                self.sizes.append(len(texts))
                return [[0.0, 1.0] for _ in texts]

            def __call__(self, text: str) -> list[float]:
                raise AssertionError("embed_text should not run when embed_texts exists")

        batch = _Batch()
        limit = 3
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=limit,
            query_vector=[0.0, 1.0],
            embed=batch,
        )
        cap = embed_candidate_cap(limit)
        self.assertEqual(batch.calls, 1)
        self.assertEqual(batch.sizes, [cap])
        self.assertEqual(len(loaded), limit)


class _Record:
    def __init__(self, data: dict):
        self.data = data


class StructuredRecordTests(unittest.TestCase):
    def test_drops_structured_text_that_only_matches_an_older_copy(self):
        older = TextChunk(
            id="old",
            text="The cedar signal code is 4410-OTTER.",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 3, 57),
        )
        newest = TextChunk(
            id="new",
            text="Sync probe cedar was removed.",
            metadata={"source_key": "src-1"},
            inserted_at=datetime.datetime(2026, 11, 16, 7, 13, 22),
        )
        store = _Store([older, newest])
        records = [
            _Record({"text": older.text, "mode": "triples"}),
            _Record({"text": newest.text, "mode": "triples"}),
            _Record({"text": "other note", "mode": "note"}),
        ]
        current = "\n".join(
            current_record_texts(
                store,
                "brain",
                records,
                [older, newest],
                include_history=False,
            )
        )
        self.assertNotIn("4410-OTTER", current)
        self.assertIn("was removed", current)
        self.assertIn("other note", current)
        history = "\n".join(
            current_record_texts(
                store,
                "brain",
                records,
                [older, newest],
                include_history=True,
            )
        )
        self.assertIn("4410-OTTER", history)


if __name__ == "__main__":
    unittest.main()
