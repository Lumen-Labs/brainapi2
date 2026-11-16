from __future__ import annotations

import unittest

from evals.baseline.contradict import detect_conflicts
from evals.baseline.index import HASH_INDEX_CONFIG, build_index
from evals.datasets.loader import docs_for_run, load_corpus, load_gold_pairs, pairs_for_run
from evals.metrics import conflict_prf, pair_key


class ConflictTests(unittest.TestCase):
    def test_seeded_pairs_recovered(self) -> None:
        docs = docs_for_run(load_corpus("policy_conflicts"), smoke=True)
        gold = [
            pair_key(a, b)
            for a, b in pairs_for_run(
                load_gold_pairs("policy_conflicts"), docs, smoke=True
            )
        ]
        index = build_index(
            docs, {**HASH_INDEX_CONFIG, "retrieve_k": 12, "rerank_depth": 8}
        )
        detected = detect_conflicts(index, docs, threshold=0.5, neighbors=10)
        pred = [pair_key(row["left"], row["right"]) for row in detected]
        prf = conflict_prf(pred, gold)
        self.assertGreaterEqual(prf["recall"], 0.75, msg=pred)
        self.assertGreaterEqual(prf["precision"], 0.75, msg=pred)

    def test_plan_scoped_sla_not_a_pair(self) -> None:
        docs = docs_for_run(load_corpus("policy_conflicts"), smoke=True)
        index = build_index(
            docs, {**HASH_INDEX_CONFIG, "retrieve_k": 12, "rerank_depth": 8}
        )
        detected = detect_conflicts(index, docs, threshold=0.5, neighbors=10)
        pred = {pair_key(row["left"], row["right"]) for row in detected}
        self.assertNotIn(pair_key("stmt-sla-std", "stmt-sla-ent"), pred)
