import unittest

from src.constants.kg import Node, Predicate
from src.core.saving.conflicts import (
    IDENTITY_FUNCTIONAL_PREDICATES,
    detect_conflicts,
    find_conflict_candidates,
    is_functional_predicate,
    snapshot_from_triple,
    values_equivalent,
)
from src.core.saving.identity import stamp_provenance
from src.core.saving.source_passages import default_text_source_key
from src.core.saving.validity_cardinality import (
    SINGLE_VALUED_RELATIONSHIP_TYPES,
    should_close_outgoing_edge,
)


class MemoryConflictStore:
    def __init__(self, chunks=None):
        self.rows = {}
        self.chunks = chunks or {}
        self.cursors = {}

    def save_conflict(self, conflict, brain_id):
        self.rows[conflict.id] = conflict
        return conflict

    def update_conflict(
        self, conflict, brain_id, if_status=None, if_resolution_method=None
    ):
        existing = self.rows.get(conflict.id)
        if if_status is not None and (existing is None or existing.status != if_status):
            return None
        if if_resolution_method is not None:
            method = ""
            if existing is not None and existing.resolution is not None:
                method = existing.resolution.method or ""
            if method != if_resolution_method:
                return None
        self.rows[conflict.id] = conflict
        return conflict

    def get_conflict_by_id(self, id, brain_id):
        return self.rows.get(id)

    def get_conflict_by_pair_key(self, pair_key, brain_id):
        for row in self.rows.values():
            if row.pair_key == pair_key:
                return row
        return None

    def get_conflicts_list(
        self,
        brain_id,
        limit=10,
        skip=0,
        status=None,
        source=None,
        subject=None,
        since=None,
        until=None,
    ):
        items = list(self.rows.values())
        if status:
            items = [item for item in items if item.status == status]
        total = len(items)
        return items[skip : skip + limit], total

    def get_open_conflicts_for_facts(self, fact_ids, brain_id):
        wanted = set(fact_ids)
        return [
            item
            for item in self.rows.values()
            if item.status == "open" and wanted.intersection(item.fact_ids())
        ]

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        return [self.chunks[item] for item in ids if item in self.chunks], []

    def get_conflict_scan_cursor(self, brain_id):
        return self.cursors.get(brain_id)

    def set_conflict_scan_cursor(self, brain_id, cursor):
        if not cursor:
            self.cursors.pop(brain_id, None)
        else:
            self.cursors[brain_id] = dict(cursor)


class FakeGraph:
    def __init__(self, triples):
        self.triples = triples
        self.calls = []

    def list_validity_relationships(
        self,
        brain_id,
        skip=0,
        limit=100,
        since=None,
        until=None,
        subject_uuids=None,
        relationship_uuids=None,
        after_subject_uuid=None,
        after_predicate=None,
        after_relationship_uuid=None,
    ):
        from src.core.saving.conflicts import (
            _sort_validity_triples,
            _triples_after_cursor,
        )

        self.calls.append(
            {
                "brain_id": brain_id,
                "skip": skip,
                "limit": limit,
                "subject_uuids": list(subject_uuids) if subject_uuids else None,
                "relationship_uuids": (
                    list(relationship_uuids) if relationship_uuids else None
                ),
                "after_subject_uuid": after_subject_uuid,
                "after_predicate": after_predicate,
                "after_relationship_uuid": after_relationship_uuid,
            }
        )
        items = self.triples
        if subject_uuids:
            allowed = set(subject_uuids)
            items = [item for item in items if item[0].uuid in allowed]
        if relationship_uuids:
            allowed_rels = set(relationship_uuids)
            items = [item for item in items if item[1].uuid in allowed_rels]
        ordered = _triples_after_cursor(
            _sort_validity_triples(items),
            after_subject_uuid=after_subject_uuid,
            after_predicate=after_predicate,
            after_relationship_uuid=after_relationship_uuid,
        )
        return ordered[skip : skip + limit], len(items)


TEXT_A = (
    "QARC-20261126 Halvorsen Dynamics company profile (published 2024-03-01). "
    "Halvorsen Dynamics is a marine robotics company headquartered in Bergen, Norway. "
    "As of 2024-03-01, the CEO of Halvorsen Dynamics is Ingrid Solheim. "
    "The Halvorsen Dynamics Tidewalker drone is priced at 12,400 EUR per unit, "
    "effective 2024-03-01. Halvorsen Dynamics has 240 employees and was founded "
    "in 2011 by Ingrid Solheim and Pavel Okonkwo. Its main factory is in Stavanger "
    "and produces 900 Tidewalker drones per year."
)
TEXT_B = (
    "QARC-20261126 Halvorsen Dynamics press release (published 2025-06-15). "
    "Effective 2025-06-15, the CEO of Halvorsen Dynamics is Marcus Ekberg, "
    "who replaced Ingrid Solheim. The Halvorsen Dynamics Tidewalker drone is "
    "now priced at 14,900 EUR per unit, effective 2025-06-15. Halvorsen Dynamics "
    "is headquartered in Oslo, Norway. Halvorsen Dynamics was founded in 2013. "
    "Halvorsen Dynamics opened a second factory in Trondheim."
)
TEXT_C = (
    "QARC-20261126 Corvella Teahouse notes (published 2025-01-10). "
    "Corvella Teahouse is a cafe chain based in Lucca, Italy, owned by "
    "Benedetta Raffaeli. Corvella Teahouse operates 17 locations in Tuscany. "
    "Its signature drink, the Saffron Mist latte, costs 4.50 EUR. Corvella "
    "Teahouse sources its tea leaves from a farm in Darjeeling run by Arjun Thapa. "
    "Update 2025-09-01: Corvella Teahouse now operates 22 locations, and the "
    "Saffron Mist latte costs 5.20 EUR."
)


def _node(uuid, name, labels=None):
    return Node(uuid=uuid, name=name, labels=labels or ["ENTITY"], properties={})


def _pred(uuid, name, props):
    return Predicate(uuid=uuid, name=name, description="", properties=dict(props))


def _fact(subject, fact_id, predicate, obj, **props):
    return (subject, _pred(fact_id, predicate, props), obj)


def _source_key(text, filename=None):
    return default_text_source_key(text, filename)


def _halvorsen_triples(*, distinct_subject_uuids=False, close_dated_windows=False):
    if distinct_subject_uuids:
        org_a = _node("org-a", "Halvorsen Dynamics", ["ORG", "COMPANY"])
        org_b = _node("org-b", "Halvorsen Dynamics", ["ORG", "COMPANY"])
    else:
        org_a = org_b = _node("org-hd", "Halvorsen Dynamics", ["ORG", "COMPANY"])
    cafe = _node("org-ct", "Corvella Teahouse", ["ORG", "COMPANY"])
    bergen = _node("city-bergen", "Bergen")
    oslo = _node("city-oslo", "Oslo")
    ingrid = _node("person-ingrid", "Ingrid Solheim", ["PERSON"])
    marcus = _node("person-marcus", "Marcus Ekberg", ["PERSON"])
    stavanger = _node("city-stavanger", "Stavanger")
    trondheim = _node("city-trondheim", "Trondheim")
    lucca = _node("city-lucca", "Lucca")
    key_a = _source_key(TEXT_A, "profile.txt")
    key_b = _source_key(TEXT_B, "press.txt")
    key_c = _source_key(TEXT_C)
    ceo_a_to = "2025-06-15T00:00:00Z" if close_dated_windows else None
    price_a_to = "2025-06-15T00:00:00Z" if close_dated_windows else None
    loc_c_to = "2025-09-01T00:00:00Z" if close_dated_windows else None
    latte_c_to = "2025-09-01T00:00:00Z" if close_dated_windows else None

    def props(key, valid_from, valid_to=None):
        payload = {"source_key": key, "valid_from": valid_from}
        if valid_to:
            payload["valid_to"] = valid_to
        return payload

    triples = [
        _fact(
            org_a,
            "hq-bergen",
            "HEADQUARTERED_IN",
            bergen,
            **props(key_a, "2024-03-01T00:00:00Z"),
        ),
        _fact(
            org_b,
            "hq-oslo",
            "HEADQUARTERED_IN",
            oslo,
            **props(key_b, "2025-06-15T00:00:00Z"),
        ),
        _fact(
            org_a,
            "founded-2011",
            "FOUNDED_IN",
            _node("year-2011", "2011"),
            **props(key_a, "2024-03-01T00:00:00Z"),
        ),
        _fact(
            org_b,
            "founded-2013",
            "FOUNDED_IN",
            _node("year-2013", "2013"),
            **props(key_b, "2025-06-15T00:00:00Z"),
        ),
        _fact(
            org_a,
            "ceo-ingrid",
            "HAS_CEO",
            ingrid,
            **props(key_a, "2024-03-01T00:00:00Z", ceo_a_to),
        ),
        _fact(
            org_b,
            "ceo-marcus",
            "HAS_CEO",
            marcus,
            **props(key_b, "2025-06-15T00:00:00Z"),
        ),
        _fact(
            org_a,
            "price-12400",
            "PRICE",
            _node("price-old", "12400 EUR"),
            **props(key_a, "2024-03-01T00:00:00Z", price_a_to),
        ),
        _fact(
            org_b,
            "price-14900",
            "PRICE",
            _node("price-new", "14900 EUR"),
            **props(key_b, "2025-06-15T00:00:00Z"),
        ),
        _fact(
            org_a,
            "factory-stavanger",
            "HAS_FACTORY",
            stavanger,
            **props(key_a, "2024-03-01T00:00:00Z"),
        ),
        _fact(
            org_b,
            "factory-trondheim",
            "HAS_FACTORY",
            trondheim,
            **props(key_b, "2025-06-15T00:00:00Z"),
        ),
        _fact(
            cafe,
            "based-lucca",
            "BASED_IN",
            lucca,
            **props(key_c, "2025-01-10T00:00:00Z"),
        ),
        _fact(
            cafe,
            "locations-17",
            "OPERATES",
            _node("loc-17", "17 locations"),
            **props(key_c, "2025-01-10T00:00:00Z", loc_c_to),
        ),
        _fact(
            cafe,
            "locations-22",
            "OPERATES",
            _node("loc-22", "22 locations"),
            **props(key_c, "2025-01-10T00:00:00Z")
            if not close_dated_windows
            else {
                "source_key": key_c,
                "valid_from": "2025-09-01T00:00:00Z",
            },
        ),
        _fact(
            cafe,
            "latte-450",
            "COSTS",
            _node("latte-old", "4.50 EUR"),
            **props(key_c, "2025-01-10T00:00:00Z", latte_c_to),
        ),
        _fact(
            cafe,
            "latte-520",
            "COSTS",
            _node("latte-new", "5.20 EUR"),
            **props(key_c, "2025-01-10T00:00:00Z")
            if not close_dated_windows
            else {
                "source_key": key_c,
                "valid_from": "2025-09-01T00:00:00Z",
            },
        ),
    ]
    return triples


def _pairs(found):
    return {tuple(sorted(item.fact_ids())) for item in found}


class HalvorsenConflictScanTests(unittest.TestCase):
    def test_paste_path_assigns_stable_source_keys(self):
        key_a = _source_key(TEXT_A, "profile.txt")
        key_b = _source_key(TEXT_B, "press.txt")
        key_c = _source_key(TEXT_C)
        self.assertEqual(key_a, "text:profile.txt")
        self.assertEqual(key_b, "text:press.txt")
        self.assertEqual(key_c, "text:paste")
        self.assertEqual(key_a, _source_key("edited " + TEXT_A, "profile.txt"))
        self.assertEqual(
            default_text_source_key(TEXT_A, "profile.txt", source_id="qa-profile"),
            "qa-profile",
        )

    def test_identity_functional_predicates_are_recognized(self):
        self.assertEqual(IDENTITY_FUNCTIONAL_PREDICATES, SINGLE_VALUED_RELATIONSHIP_TYPES)
        self.assertTrue("HEADQUARTERED_IN" in IDENTITY_FUNCTIONAL_PREDICATES)
        self.assertFalse("LOCATED_IN" in IDENTITY_FUNCTIONAL_PREDICATES)
        self.assertFalse("WORKS_AT" in IDENTITY_FUNCTIONAL_PREDICATES)
        self.assertTrue(is_functional_predicate("HEADQUARTERED_IN", ["Bergen", "Oslo"]))
        self.assertTrue(is_functional_predicate("FOUNDED_IN", ["2011", "2013"]))
        self.assertTrue(is_functional_predicate("HAS_CEO", ["Ingrid Solheim", "Marcus Ekberg"]))
        self.assertTrue(is_functional_predicate("PRICE", ["12400 EUR", "14900 EUR"]))
        self.assertTrue(is_functional_predicate("LOCATED_IN", ["Milan", "Italy"]))
        self.assertFalse(is_functional_predicate("HAS_FACTORY", ["Stavanger", "Trondheim"]))
        self.assertFalse(is_functional_predicate("WORKS_AT", ["Acme", "Globex"]))
        self.assertFalse(is_functional_predicate("FOUNDED", ["Acme", "Globex"]))
        self.assertFalse(is_functional_predicate("HAS_TITLE", ["CEO", "Director"]))
        self.assertFalse(is_functional_predicate("OPERATES", ["17 locations", "22 locations"]))
        self.assertFalse(values_equivalent("2011", "2013"))
        self.assertTrue(values_equivalent("2011", "2011"))

    def test_ingest_does_not_close_headquarters_across_sources(self):
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={"source_key": _source_key(TEXT_A, "profile.txt")},
                incoming_source_key=_source_key(TEXT_B, "press.txt"),
                require_same_source=True,
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={"source_key": "doc-a"},
                incoming_source_key="doc-a",
                require_same_source=True,
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={},
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={},
                incoming_source_key=_source_key(TEXT_B, "press.txt"),
                require_same_source=True,
            )
        )

    def test_stamp_provenance_keeps_source_key_on_text_ingest_facts(self):
        key = _source_key(TEXT_A, "profile.txt")
        props = stamp_provenance(
            {},
            source_chunk_id="chunk-a",
            source_timestamp="2024-03-01T00:00:00Z",
            source_key=key,
        )
        self.assertEqual(props["source_key"], key)
        self.assertEqual(props["source_chunk_ids"], ["chunk-a"])
        kept = stamp_provenance({"source_key": key}, source_chunk_id="chunk-b")
        self.assertEqual(kept["source_key"], key)

    def test_deterministic_scan_finds_true_conflicts(self):
        found = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(_halvorsen_triples()),
            "brain",
            use_llm=False,
        )
        pairs = _pairs(found)
        self.assertIn(("hq-bergen", "hq-oslo"), pairs)
        self.assertIn(("founded-2011", "founded-2013"), pairs)
        self.assertNotIn(("factory-stavanger", "factory-trondheim"), pairs)
        kinds = {
            tuple(sorted(item.fact_ids())): item.disagreement_type for item in found
        }
        self.assertEqual(kinds[("hq-bergen", "hq-oslo")], "value_mismatch")

    def test_deterministic_scan_pairs_unmerged_halvorsen_mentions_by_name(self):
        found = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(_halvorsen_triples(distinct_subject_uuids=True)),
            "brain",
            use_llm=False,
        )
        pairs = _pairs(found)
        self.assertIn(("hq-bergen", "hq-oslo"), pairs)
        self.assertIn(("founded-2011", "founded-2013"), pairs)

    def test_overlapping_dated_facts_are_conflicts_closed_windows_are_not(self):
        overlapping = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(_halvorsen_triples()),
            "brain",
            use_llm=False,
        )
        overlapping_pairs = _pairs(overlapping)
        self.assertIn(("ceo-ingrid", "ceo-marcus"), overlapping_pairs)
        self.assertIn(("price-12400", "price-14900"), overlapping_pairs)
        self.assertNotIn(("locations-17", "locations-22"), overlapping_pairs)
        self.assertNotIn(("latte-450", "latte-520"), overlapping_pairs)

        closed = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(_halvorsen_triples(close_dated_windows=True)),
            "brain",
            use_llm=False,
        )
        closed_pairs = _pairs(closed)
        self.assertIn(("hq-bergen", "hq-oslo"), closed_pairs)
        self.assertIn(("founded-2011", "founded-2013"), closed_pairs)
        self.assertNotIn(("ceo-ingrid", "ceo-marcus"), closed_pairs)
        self.assertNotIn(("price-12400", "price-14900"), closed_pairs)
        self.assertNotIn(("locations-17", "locations-22"), closed_pairs)
        self.assertNotIn(("latte-450", "latte-520"), closed_pairs)

    def test_headquarters_is_definite_not_semantic(self):
        snapshots = [
            snapshot_from_triple(item)
            for item in _halvorsen_triples()
            if item[1].uuid in {"hq-bergen", "hq-oslo"}
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(semantic, [])
        self.assertEqual(len(definite), 1)
        self.assertEqual(definite[0][2], "value_mismatch")

    def test_located_in_milan_vs_italy_is_not_definite(self):
        acme = _node("org-acme", "Acme", ["ORG", "COMPANY"])
        snapshots = [
            snapshot_from_triple(
                _fact(
                    acme,
                    "loc-milan",
                    "LOCATED_IN",
                    _node("city-milan", "Milan"),
                    source_key="wiki",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ),
            snapshot_from_triple(
                _fact(
                    acme,
                    "loc-italy",
                    "LOCATED_IN",
                    _node("country-italy", "Italy"),
                    source_key="press",
                    valid_from="2024-06-01T00:00:00Z",
                )
            ),
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(len(semantic), 1)

    def test_two_jobs_are_not_a_conflict(self):
        person = _node("person-a", "Ada Lovelace", ["PERSON"])
        snapshots = [
            snapshot_from_triple(
                _fact(
                    person,
                    "job-acme",
                    "WORKS_AT",
                    _node("org-acme", "Acme"),
                    source_key="resume",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ),
            snapshot_from_triple(
                _fact(
                    person,
                    "job-globex",
                    "WORKS_AT",
                    _node("org-globex", "Globex"),
                    source_key="press",
                    valid_from="2024-06-01T00:00:00Z",
                )
            ),
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(semantic, [])
        found = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(
                [
                    _fact(
                        person,
                        "job-acme",
                        "WORKS_AT",
                        _node("org-acme", "Acme"),
                        source_key="resume",
                        valid_from="2024-01-01T00:00:00Z",
                    ),
                    _fact(
                        person,
                        "job-globex",
                        "WORKS_AT",
                        _node("org-globex", "Globex"),
                        source_key="press",
                        valid_from="2024-06-01T00:00:00Z",
                    ),
                ]
            ),
            "brain",
            use_llm=False,
        )
        self.assertEqual(list(found), [])

    def test_john_smith_nodes_with_different_labels_are_not_paired(self):
        person = _node("person-js", "John Smith", ["PERSON"])
        company = _node("org-js", "John Smith", ["ORG", "COMPANY"])
        snapshots = [
            snapshot_from_triple(
                _fact(
                    person,
                    "hq-person",
                    "HEADQUARTERED_IN",
                    _node("city-oslo", "Oslo"),
                    source_key="bio",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ),
            snapshot_from_triple(
                _fact(
                    company,
                    "hq-org",
                    "HEADQUARTERED_IN",
                    _node("city-bergen", "Bergen"),
                    source_key="press",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ),
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(semantic, [])

        org = _node("org-a", "Halvorsen Dynamics", ["ORG"])
        company = _node("org-b", "Halvorsen Dynamics", ["COMPANY"])
        compatible = [
            snapshot_from_triple(
                _fact(
                    org,
                    "hq-bergen",
                    "HEADQUARTERED_IN",
                    _node("city-bergen", "Bergen"),
                    source_key="profile",
                    valid_from="2024-03-01T00:00:00Z",
                )
            ),
            snapshot_from_triple(
                _fact(
                    company,
                    "hq-oslo",
                    "HEADQUARTERED_IN",
                    _node("city-oslo", "Oslo"),
                    source_key="press",
                    valid_from="2025-06-15T00:00:00Z",
                )
            ),
        ]
        definite_ok, semantic_ok = find_conflict_candidates(compatible)
        self.assertEqual(semantic_ok, [])
        self.assertEqual(len(definite_ok), 1)

    def test_edited_repaste_closes_old_edges_instead_of_conflicting(self):
        old_key = _source_key("Halvorsen is headquartered in Bergen.", "note.txt")
        new_key = _source_key("Halvorsen is headquartered in Oslo.", "note.txt")
        self.assertEqual(old_key, new_key)
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={"source_key": old_key},
                incoming_source_key=new_key,
                require_same_source=True,
            )
        )
        org = _node("org-hd", "Halvorsen Dynamics", ["ORG", "COMPANY"])
        snapshots = [
            snapshot_from_triple(
                _fact(
                    org,
                    "hq-bergen",
                    "HEADQUARTERED_IN",
                    _node("city-bergen", "Bergen"),
                    source_key=old_key,
                    valid_from="2024-03-01T00:00:00Z",
                )
            ),
            snapshot_from_triple(
                _fact(
                    org,
                    "hq-oslo",
                    "HEADQUARTERED_IN",
                    _node("city-oslo", "Oslo"),
                    source_key=new_key,
                    valid_from="2025-06-15T00:00:00Z",
                )
            ),
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(semantic, [])


if __name__ == "__main__":
    unittest.main()
