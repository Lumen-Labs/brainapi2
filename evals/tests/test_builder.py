from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.datasets.build_public_corpora import (
    fetch_far_xml,
    is_natural_clash,
    mutate_statement,
    parse_far_sections,
    phrase_for_gold,
    repair_knowhow_questions,
)


class BuilderTests(unittest.TestCase):
    def test_parse_far_keeps_inline_emphasis_text(self) -> None:
        xml = """<?xml version="1.0"?>
        <ECFR>
          <DIV8 TYPE="SECTION" N="52.232-1">
            <HEAD>52.232-1 Payments.</HEAD>
            <P>The Contractor shall be paid within <E T="03">30</E> days after delivery of accepted supplies.</P>
          </DIV8>
        </ECFR>
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "far.xml"
            path.write_text(xml, encoding="utf-8")
            rows = parse_far_sections(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("30", rows[0]["text"])
        self.assertIn("days after delivery", rows[0]["text"])

    def test_mutate_replaces_matched_span_not_id_digits(self) -> None:
        text = "[stmt-tt-062bcbff67] Repos must use Creative Commons 0, or CC0."
        mutated = mutate_statement(text)
        self.assertIsNone(mutated)

        billed = "[stmt-gi-24aa8d4da6] The amount usually ranges from $1 to the cost."
        result = mutate_statement(billed)
        self.assertIsNotNone(result)
        new_text, rationale = result
        self.assertTrue(new_text.startswith("[stmt-gi-24aa8d4da6]"))
        self.assertIn("$8", new_text)
        self.assertIn("$1 vs $8", rationale)

    def test_skip_far_fetch_does_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = fetch_far_xml("2018-01-03", "2", Path(tmp), skip_fetch=True)
        self.assertIsNone(path)

    def test_natural_clash_rejects_generic_topic_and_mixed_units(self) -> None:
        left = {
            "source": "github_docs",
            "title": "About mandatory two-factor authentication",
            "text": "Enable 2FA within the 45 day setup period.",
        }
        refund = {
            "source": "github_docs",
            "title": "FAQ about changes to GitHub's plans",
            "text": "These refunds will be processed in approximately 30 days. Plans include $12 of compute.",
        }
        hold_amount = {
            "source": "github_docs",
            "title": "Troubleshooting authorization holds",
            "text": "The amount usually ranges from $1 to the cost of the service.",
        }
        hold_days = {
            "source": "github_docs",
            "title": "Troubleshooting authorization holds",
            "text": "If funds remain unavailable after 10 business days, contact the bank.",
        }
        auth_peer = {
            "source": "github_docs",
            "title": "Configuring two-factor authentication",
            "text": "Accounts must enable 2FA within 28 days.",
        }
        self.assertFalse(is_natural_clash(left, refund))
        self.assertFalse(is_natural_clash(hold_amount, hold_days))
        self.assertTrue(is_natural_clash(left, auth_peer))

    def test_repair_knowhow_uses_hop_exclusive_spans(self) -> None:
        docs = [
            {
                "id": "rb-a",
                "text": "Mitigation: Cordon and drain node to migrate apps from the failing host.",
            },
            {
                "id": "rb-b",
                "text": "Diagnosis: Check resource usage for the namespace in kubectl describe quota.",
            },
        ]
        questions = [
            {
                "id": "q1",
                "split": "test",
                "hops": ["rb-a", "rb-b"],
                "must_include": ["Talk with developers or read documentation about the", "shared"],
                "gold": "old",
            }
        ]
        repaired = repair_knowhow_questions(docs, questions)
        self.assertEqual(len(repaired), 1)
        left, right = repaired[0]["must_include"]
        self.assertIn(left.lower(), docs[0]["text"].lower())
        self.assertNotIn(left.lower(), docs[1]["text"].lower())
        self.assertIn(right.lower(), docs[1]["text"].lower())
        self.assertNotIn(right.lower(), docs[0]["text"].lower())
        self.assertFalse(left.lower().endswith(" the"))
        self.assertGreaterEqual(len(phrase_for_gold(left).split()), 6)
