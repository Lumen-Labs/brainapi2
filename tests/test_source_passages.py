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
        self.marked: list[tuple[list[str], str, str | None, str | None]] = []
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

    def mark_text_chunks_superseded(
        self, ids, superseded_at, brain_id, superseded_by=None, valid_to=None
    ):
        self.marked.append((list(ids), superseded_at, superseded_by, valid_to))
        for chunk in self.chunks:
            if str(chunk.id) in ids:
                chunk.metadata["superseded_at"] = superseded_at
                if valid_to:
                    chunk.metadata["valid_to"] = valid_to
                if superseded_by:
                    chunk.metadata["superseded_by"] = superseded_by

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
        self.assertEqual(store.marked[0][0], ["c1"])
        self.assertEqual(store.marked[0][1], "16/11/2026")
        self.assertIsNotNone(store.marked[0][3])
        self.assertEqual(previous.metadata["superseded_at"], "16/11/2026")
        self.assertTrue(previous.metadata.get("valid_to"))
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
        self.assertIsNotNone(store.marked[0][3])

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
                        "superseded_at": f"2026-11-16T12:{index:02d}:00",
                    },
                    inserted_at=datetime.datetime(2026, 11, 16, 12, index, 0),
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
        original = "[superseded] revision body 0 unique snack"
        newest = {
            f"[superseded] cedar signal {index}" for index in range(35, 40)
        }
        self.assertEqual(cap, 6)
        self.assertEqual(embed_candidate_cap(200), 500)
        self.assertLessEqual(len(calls), cap)
        self.assertGreater(len(calls), 0)
        self.assertIn(original, calls)
        self.assertTrue(newest <= set(calls))
        self.assertEqual(len(loaded), limit)
        self.assertTrue(all("cedar" in body for body in loaded))
        self.assertTrue(store.limits)
        self.assertLess(max(store.limits), 500)
        self.assertLessEqual(max(store.limits), cap + 16)

    def test_prior_revision_keeps_original_when_over_cap(self):
        older = []
        for index in range(40):
            text = (
                "The original cedar signal code is 4410-OTTER."
                if index == 0
                else f"revision body {index} unique snack"
            )
            older.append(
                TextChunk(
                    id=f"old-{index}",
                    text=text,
                    metadata={
                        "source_key": "src-1",
                        "superseded_at": f"2026-11-16T12:{index:02d}:00",
                    },
                    inserted_at=datetime.datetime(2026, 11, 16, 12, index, 0),
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
            return [1.0, 0.0] if "4410-OTTER" in text else [0.0, 1.0]

        limit = 1
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=limit,
            query_text="what was the original cedar signal code",
            query_vector=[1.0, 0.0],
            embed=embed,
        )
        cap = embed_candidate_cap(limit)
        original = "[superseded] The original cedar signal code is 4410-OTTER."
        self.assertEqual(cap, 3)
        self.assertLess(len(older), 500)
        self.assertGreater(len(older), cap)
        self.assertLessEqual(len(calls), cap)
        self.assertIn(original, calls)
        self.assertEqual(loaded, [original])
        self.assertLess(max(store.limits), 500)

    def test_prior_revision_pages_past_current_chunks(self):
        currents = [
            TextChunk(
                id=f"live-{index}",
                text=f"live copy {index}",
                metadata={"source_key": "src-1"},
                inserted_at=datetime.datetime(2026, 11, 17, 0, index, 0),
            )
            for index in range(50)
        ]
        older = [
            TextChunk(
                id=f"old-{index}",
                text=f"cedar signal {index}",
                metadata={
                    "source_key": "src-1",
                    "superseded_at": f"2026-11-16T12:{index:02d}:00",
                },
                inserted_at=datetime.datetime(2026, 11, 16, 12, index, 0),
            )
            for index in range(5)
        ]
        live = currents[-1]
        store = _Store(currents + older)
        loaded = prior_revision_texts(
            store,
            "brain",
            [("live-49", 0.2, live.text)],
            limit=2,
            query_text="cedar signal",
        )
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all("cedar signal" in body for body in loaded))

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
                    "superseded_at": f"2026-11-16T12:{index:02d}:00",
                },
                inserted_at=datetime.datetime(2026, 11, 16, 12, index, 0),
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
                self.singles: list[str] = []

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                self.sizes.append(len(texts))
                return [[0.0, 1.0] for _ in texts]

            def __call__(self, text: str) -> list[float]:
                self.singles.append(text)
                return [0.0, 1.0]

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
        self.assertEqual(batch.singles, [])
        self.assertEqual(len(loaded), limit)

    def test_prior_revision_skips_embed_without_query_vector(self):
        older = TextChunk(
            id="old",
            text="The cedar signal code is 4410-OTTER.",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T07:11:00",
            },
        )
        live = TextChunk(id="new", text="live", metadata={"source_key": "src-1"})
        store = _Store([older, live])
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0]

        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
            query_text="cedar signal",
            embed=embed,
        )
        self.assertEqual(calls, [])
        self.assertEqual(loaded, ["[superseded] The cedar signal code is 4410-OTTER."])

    def test_prior_revision_does_not_mix_cosine_and_lexical(self):
        snack = TextChunk(
            id="snack",
            text="office pretzels",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T07:00:00",
                "embeddings": [1.0, 0.0],
            },
        )
        cedar = TextChunk(
            id="cedar",
            text="cedar signal",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T08:00:00",
            },
        )
        live = TextChunk(id="new", text="live", metadata={"source_key": "src-1"})
        store = _Store([snack, cedar, live])
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
            query_text="",
            query_vector=[1.0, 0.0],
        )
        self.assertEqual(loaded, ["[superseded] cedar signal"])

    def test_prior_revision_keeps_newest_duplicate_body(self):
        older = TextChunk(
            id="old",
            text="same body",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T10:00:00",
            },
        )
        newer = TextChunk(
            id="newer",
            text="same body",
            metadata={
                "source_key": "src-1",
                "superseded_at": "2026-11-16T12:00:00",
            },
        )
        live = TextChunk(id="new", text="live", metadata={"source_key": "src-1"})
        store = _Store([older, newer, live])
        loaded = prior_revision_texts(
            store,
            "brain",
            [("new", 0.2, live.text)],
            limit=1,
        )
        self.assertEqual(loaded, ["[superseded] same body"])


class _Record:
    def __init__(self, data: dict, inserted_at=None):
        self.data = data
        self.inserted_at = inserted_at


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

    def test_as_of_drops_records_inserted_after_the_snapshot(self):
        chunk = TextChunk(
            id="c1",
            text="keep me",
            metadata={"valid_from": "2024-01-01T00:00:00Z"},
        )
        records = [
            _Record(
                {"text": "future note"},
                inserted_at=datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc),
            ),
            _Record(
                {"text": "past note"},
                inserted_at=datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc),
            ),
        ]
        lines = current_record_texts(
            _Store([chunk]),
            "brain",
            records,
            [chunk],
            include_history=False,
            as_of=datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc),
        )
        joined = "\n".join(lines)
        self.assertIn("past note", joined)
        self.assertNotIn("future note", joined)


class AsOfPassageTests(unittest.TestCase):
    def test_as_of_returns_the_copy_valid_at_that_instant(self):
        v1 = TextChunk(
            id="v1",
            text="Clause 12: retain records for 5 years.",
            metadata={
                "source_key": "reg-12",
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": "2025-01-01T00:00:00Z",
                "superseded_at": "2025-01-01T00:00:00Z",
            },
        )
        v2 = TextChunk(
            id="v2",
            text="Clause 12: retain records for 7 years.",
            metadata={
                "source_key": "reg-12",
                "valid_from": "2025-01-01T00:00:00Z",
            },
        )
        store = _Store([v1, v2])
        hits = [("v1", 0.1, v1.text), ("v2", 0.2, v2.text)]
        mid = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc)
        after = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)
        at_v1 = screen_passage_hits(
            store, "brain", hits, include_history=False, as_of=mid
        )
        at_v2 = screen_passage_hits(
            store, "brain", hits, include_history=False, as_of=after
        )
        self.assertEqual([item[2] for item in at_v1], [v1.text])
        self.assertEqual([item[2] for item in at_v2], [v2.text])
        self.assertFalse(any(item[2].startswith("[superseded]") for item in at_v1))

    def test_as_of_omits_hits_when_lookup_fails(self):
        class Broken(_Store):
            def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
                raise RuntimeError("lookup down")

        v2 = TextChunk(
            id="v2",
            text="seven years",
            metadata={"valid_from": "2025-01-01T00:00:00Z"},
        )
        store = Broken([v2])
        hits = [("v2", 0.2, v2.text)]
        as_of = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc)
        self.assertEqual(
            screen_passage_hits(store, "brain", hits, include_history=False, as_of=as_of),
            [],
        )
        self.assertEqual(
            screen_passage_hits(store, "brain", hits, include_history=False),
            hits,
        )

    def test_as_of_facts_batch_lookups_and_drop_unverified(self):
        class Capped(_Store):
            def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
                self.calls = getattr(self, "calls", [])
                self.calls.append(list(ids))
                return super().get_text_chunks_by_ids(ids[:80], with_observations, brain_id)

        future = TextChunk(
            id="future",
            text="too new",
            metadata={"valid_from": "2025-01-01T00:00:00Z"},
        )
        store = Capped([future])
        facts = [{"id": f"f{i}", "chunk_ids": [f"missing-{i}"]} for i in range(80)]
        facts.append({"id": "f-future", "chunk_ids": ["future"]})
        as_of = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc)
        kept = current_fact_candidates(
            store, "brain", facts, include_history=False, as_of=as_of
        )
        self.assertEqual(len(store.calls), 2)
        self.assertEqual(kept, [])

    def test_as_of_omits_undated_hits_and_live_retrieve_keeps_them(self):
        undated = TextChunk(id="plain", text="no dates", metadata={})
        dated = TextChunk(
            id="dated",
            text="dated",
            metadata={"valid_from": "2024-01-01T00:00:00Z"},
        )
        store = _Store([undated, dated])
        hits = [("plain", 0.1, undated.text), ("dated", 0.2, dated.text)]
        as_of = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc)
        self.assertEqual(
            [item[0] for item in screen_passage_hits(
                store, "brain", hits, include_history=False, as_of=as_of
            )],
            ["dated"],
        )
        self.assertEqual(
            [item[0] for item in screen_passage_hits(
                store, "brain", hits, include_history=False
            )],
            ["plain", "dated"],
        )


class SourceRevisionStampTests(unittest.TestCase):
    def test_new_revision_reuses_one_timestamp(self):
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
                source_timestamp=None,
            ),
            "insert",
        )
        self.assertNotIn("valid_from", meta)
        self.assertEqual(store.marked[0][1], meta["recorded_at"])
        self.assertEqual(store.marked[0][3], meta["recorded_at"])

    def test_source_date_does_not_become_recorded_at(self):
        previous = TextChunk(
            id="c1",
            text="hello",
            metadata={"source_key": "src-1", "source_content_hash": "old"},
        )
        store = _Store([previous])
        meta = {"source_key": "src-1"}
        recorded_at = "2026-11-18T12:00:00Z"
        self.assertEqual(
            prepare_source_revision(
                store,
                brain_id="brain",
                source_text="hello again",
                meta_keys=meta,
                source_timestamp="2020-01-01T00:00:00Z",
                recorded_at=recorded_at,
            ),
            "insert",
        )
        self.assertEqual(meta["valid_from"], "2020-01-01T00:00:00Z")
        self.assertEqual(meta["recorded_at"], recorded_at)
        self.assertNotEqual(meta["recorded_at"], meta["valid_from"])
        self.assertEqual(store.marked[0][1], "2020-01-01T00:00:00Z")
        self.assertEqual(store.marked[0][3], "2020-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
