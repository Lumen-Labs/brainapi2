from __future__ import annotations

import math
from collections import defaultdict

from evals.baseline.textlib import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = defaultdict(int)
        self.avgdl = 0.0
        self.n = 0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        counts: dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(toks))
        self.tf.append(counts)
        for tok in counts:
            self.df[tok] += 1
        self.n += 1
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        q_toks = tokenize(query)
        if not q_toks or not self.n:
            return []
        scores = [0.0] * self.n
        avgdl = self.avgdl or 1.0
        for tok in dict.fromkeys(q_toks):
            df = self.df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for i, tfs in enumerate(self.tf):
                freq = tfs.get(tok, 0)
                if not freq:
                    continue
                dl = self.doc_len[i] or 1
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[i] += idf * (freq * (self.k1 + 1.0) / denom)
        ranked = sorted(
            ((self.doc_ids[i], scores[i]) for i in range(self.n) if scores[i] > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]
