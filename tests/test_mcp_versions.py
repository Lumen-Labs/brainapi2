import unittest
from pathlib import Path

from src.services.mcp.prompt import MCP_TOOL_SPECS
from src.services.mcp.versions import (
    read_changelog,
    read_changelogs,
    read_source_versions,
)


class _Change:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, mode="json"):
        return dict(self.payload)


class _ChangelogAdapter:
    def __init__(self, rows=None, entry=None):
        self.rows = rows or []
        self.entry = entry
        self.limit = None

    def get_changelogs_list(
        self, brain_id, limit, skip, types, query_text, since=None, until=None
    ):
        self.limit = limit
        self.brain_id = brain_id
        self.types = types
        self.query_text = query_text
        self.skip = skip
        self.since = since
        self.until = until
        return self.rows

    def get_changelog_by_id(self, changelog_id, brain_id):
        self.changelog_id = changelog_id
        self.brain_id = brain_id
        return self.entry


class _Chunk:
    def __init__(self, chunk_id, text, metadata):
        self.id = chunk_id
        self.text = text
        self.metadata = metadata


class _Hit:
    def __init__(self, resource_id):
        self.metadata = {"resource_id": resource_id}


class _VersionStore:
    def __init__(self, by_key, by_id):
        self.by_key = by_key
        self.by_id = by_id
        self.searched = False

    def get_text_chunks(self, brain_id, limit, skip, metadata_eq):
        key = metadata_eq["source_key"]
        return list(self.by_key.get(key, [])), len(self.by_key.get(key, []))

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        return [self.by_id[item] for item in ids if item in self.by_id], []


class ChangelogToolTests(unittest.TestCase):
    def test_authorized_list_returns_adapter_rows_and_clamps_limit(self):
        adapter = _ChangelogAdapter(
            rows=[_Change({"id": "c1", "type": "relationship_deprecated"})]
        )
        result = read_changelogs(
            adapter,
            "brain",
            authorized=True,
            query_text="cedar",
            types=["relationship_deprecated"],
            limit=500,
            skip=2,
        )
        self.assertEqual(adapter.limit, 50)
        self.assertEqual(adapter.query_text, "cedar")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["changelogs"][0]["id"], "c1")

    def test_since_until_returns_changes_window(self):
        from src.constants.data import TextChunk

        class Store(_ChangelogAdapter):
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                if skip:
                    return [], 1
                chunk = TextChunk(
                    id="v2",
                    text="seven years",
                    metadata={"valid_from": "2025-01-01T00:00:00Z", "source_key": "reg"},
                )
                return [chunk], 1

        adapter = Store()
        result = read_changelogs(
            adapter,
            "brain",
            authorized=True,
            since="2024-12-01T00:00:00Z",
            until="2025-02-01T00:00:00Z",
        )
        self.assertEqual(result["since"], "2024-12-01T00:00:00Z")
        self.assertEqual(result["changes"][0]["id"], "v2")
        self.assertEqual(result["changes"][0]["kind"], "added")

    def test_unauthorized_list_does_not_read_the_store(self):
        adapter = _ChangelogAdapter(rows=[_Change({"id": "c1"})])
        self.assertEqual(
            read_changelogs(adapter, "brain", authorized=False),
            "Unauthorized",
        )
        self.assertIsNone(adapter.limit)

    def test_missing_changelog_is_a_not_found_string(self):
        adapter = _ChangelogAdapter(entry=None)
        self.assertEqual(
            read_changelog(adapter, "brain", "missing", authorized=True),
            "Error: changelog not found",
        )

    def test_changelog_tools_are_registered(self):
        names = {spec["name"] for spec in MCP_TOOL_SPECS}
        self.assertIn("list_changelogs", names)
        self.assertIn("get_changelog", names)
        root = Path(__file__).resolve().parents[1]
        executor = (root / "src/services/mcp/executor.py").read_text(encoding="utf-8")
        self.assertIn('tool_name == "list_changelogs"', executor)
        self.assertIn('tool_name == "get_changelog"', executor)
        self.assertIn('tool_name == "list_source_versions"', executor)
        instructions = (root / "src/services/mcp/main.py").read_text(encoding="utf-8")
        self.assertIn("HISTORY_NAVIGATION_GUIDE", instructions)


class SourceVersionToolTests(unittest.TestCase):
    def _store(self):
        older = _Chunk(
            "old",
            "The cedar signal code is 4410-OTTER.",
            {"source_key": "src-1", "superseded_at": "2026-11-16T07:11:00"},
        )
        current = _Chunk(
            "new",
            "The cedar signal code is 7720-ROBIN.",
            {"source_key": "src-1", "operation": "update"},
        )
        return _VersionStore({"src-1": [older, current]}, {"new": current, "old": older})

    def test_source_key_returns_the_sibling_search_did_not_rank(self):
        store = self._store()

        def explode(*_args, **_kwargs):
            store.searched = True
            raise AssertionError("search")

        result = read_source_versions(
            store,
            "brain",
            authorized=True,
            source_key="src-1",
            embed=explode,
            search=explode,
        )
        self.assertFalse(store.searched)
        self.assertEqual(result["sources"][0]["current"][0]["text"], "The cedar signal code is 7720-ROBIN.")
        self.assertEqual(result["sources"][0]["current"][0]["operation"], "update")
        self.assertEqual(
            result["sources"][0]["previous"][0]["text"],
            "The cedar signal code is 4410-OTTER.",
        )

    def test_query_uses_hits_only_to_discover_the_key(self):
        store = self._store()

        def embed(_text):
            return [0.1, 0.2]

        def search(_vector, _brain_id):
            store.searched = True
            return [_Hit("new")]

        result = read_source_versions(
            store,
            "brain",
            authorized=True,
            query="what did the cedar code used to be",
            embed=embed,
            search=search,
        )
        self.assertTrue(store.searched)
        previous = result["sources"][0]["previous"]
        self.assertEqual(previous[0]["id"], "old")
        self.assertEqual(previous[0]["superseded_at"], "2026-11-16T07:11:00")

    def test_empty_request_and_embedding_failure_return_errors(self):
        store = self._store()
        self.assertEqual(
            read_source_versions(store, "brain", authorized=True),
            "Error: provide source_key or query",
        )

        def embed(_text):
            raise RuntimeError("embed down")

        self.assertEqual(
            read_source_versions(
                store,
                "brain",
                authorized=True,
                query="cedar",
                embed=embed,
                search=lambda *_args: [],
            ),
            "Error: embedding search failed",
        )
        self.assertEqual(
            read_source_versions(store, "brain", authorized=False, source_key="src-1"),
            "Unauthorized",
        )

    def test_source_key_pages_previous_copies(self):
        first = _Chunk(
            "old-1",
            "code 4410",
            {"source_key": "src-1", "superseded_at": "2026-11-16T07:11:00"},
        )
        second = _Chunk(
            "old-2",
            "code 5500",
            {"source_key": "src-1", "superseded_at": "2026-11-16T08:00:00"},
        )
        current = _Chunk("new", "code 7720", {"source_key": "src-1"})
        store = _VersionStore(
            {"src-1": [first, second, current]},
            {"new": current, "old-1": first, "old-2": second},
        )
        page = read_source_versions(
            store,
            "brain",
            authorized=True,
            source_key="src-1",
            limit=1,
            skip=0,
        )
        self.assertEqual(page["skip"], 0)
        self.assertEqual(page["limit"], 1)
        self.assertEqual(page["sources"][0]["previous_total"], 2)
        self.assertEqual(page["sources"][0]["previous"][0]["id"], "old-2")
        next_page = read_source_versions(
            store,
            "brain",
            authorized=True,
            source_key="src-1",
            limit=1,
            skip=1,
        )
        self.assertEqual(next_page["sources"][0]["previous"][0]["id"], "old-1")


if __name__ == "__main__":
    unittest.main()
