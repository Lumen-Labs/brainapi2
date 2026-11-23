from __future__ import annotations

import unittest

from evals.config import PROMPT_BUDGET
from evals.prompts import ANSWER_SYSTEM, clip_context, format_passages, prompt_sha256


class PromptBudgetTests(unittest.TestCase):
    def test_budget_constants(self) -> None:
        self.assertEqual(PROMPT_BUDGET["max_passages"], 8)
        self.assertEqual(PROMPT_BUDGET["max_chars"], 6000)
        self.assertEqual(PROMPT_BUDGET["max_answer_tokens"], 256)
        self.assertEqual(PROMPT_BUDGET["temperature"], 0.0)

    def test_clip_context_respects_max(self) -> None:
        parts = ["a" * 400, "b" * 400, "c" * 400]
        clipped = clip_context(parts, 500)
        self.assertLessEqual(len(clipped), 500)

    def test_format_passages_shared(self) -> None:
        passages = [
            {"id": "1", "doc_id": "d1", "title": "t", "text": "hello " * 50}
            for _ in range(20)
        ]
        packed = format_passages(passages, max_passages=8, max_chars=600)
        self.assertLessEqual(len(packed), 600)
        self.assertIn("[1]", packed)

    def test_answer_prompt_hash_stable(self) -> None:
        self.assertEqual(len(prompt_sha256(ANSWER_SYSTEM)), 64)
        self.assertEqual(prompt_sha256(ANSWER_SYSTEM), prompt_sha256(ANSWER_SYSTEM))
