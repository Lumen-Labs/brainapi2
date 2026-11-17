from __future__ import annotations

import hashlib
import math
from typing import Protocol

from evals.baseline.textlib import tokenize
from evals.config import (
    DEFAULT_ST_EMBEDDING_MODEL,
    query_prefix_for_embedding_model,
)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def hashed_ngram_vector(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    toks = tokenize(text)
    blob = f" {' '.join(toks)} "
    for n in (2, 3, 4):
        if len(blob) < n:
            continue
        for i in range(len(blob) - n + 1):
            gram = blob[i : i + n]
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
    return _l2_normalize(vec)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class Embedder(Protocol):
    name: str

    def embed(
        self, texts: list[str], *, is_query: bool = False
    ) -> tuple[list[list[float]], dict[str, int]]:
        ...


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(
        self, texts: list[str], *, is_query: bool = False
    ) -> tuple[list[list[float]], dict[str, int]]:
        del is_query
        vectors = [hashed_ngram_vector(text, dim=self.dim) for text in texts]
        chars = sum(len(text) for text in texts)
        return vectors, {
            "embedding_calls": 0,
            "embedding_tokens": max(1, chars // 4),
            "embedding_backend": 0,
        }


class OpenAIEmbedder:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model

    def embed(
        self, texts: list[str], *, is_query: bool = False
    ) -> tuple[list[list[float]], dict[str, int]]:
        del is_query
        if not texts:
            return [], {"embedding_calls": 0, "embedding_tokens": 0, "embedding_backend": 0}
        response = self._client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return vectors, {
            "embedding_calls": 1,
            "embedding_tokens": tokens,
            "embedding_backend": 1,
        }


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str = DEFAULT_ST_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the strong hybrid baseline. "
                "Install with `pip install -r evals/requirements-strong.txt`. "
                "CI smoke uses hashed n-grams and does not need this package."
            ) from exc
        self.model_name = model or DEFAULT_ST_EMBEDDING_MODEL
        self.query_prefix = query_prefix_for_embedding_model(self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed(
        self, texts: list[str], *, is_query: bool = False
    ) -> tuple[list[list[float]], dict[str, int]]:
        if not texts:
            return [], {
                "embedding_calls": 0,
                "embedding_tokens": 0,
                "embedding_backend": 0,
            }
        payload = texts
        if is_query and self.query_prefix:
            payload = [f"{self.query_prefix}{text}" for text in texts]
        encoded = self.model.encode(
            payload,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        vectors = [list(map(float, row)) for row in encoded]
        chars = sum(len(text) for text in texts)
        return vectors, {
            "embedding_calls": 1,
            "embedding_tokens": max(1, chars // 4),
            "embedding_backend": 0,
        }


def make_embedder(
    provider: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str | None = None,
    dim: int = 256,
) -> Embedder:
    name = (provider or "sentence-transformers").strip().lower()
    if name in {"hash", "hashed", "ngram"}:
        return HashEmbedder(dim=dim)
    if name in {"openai", "azure"}:
        if not api_key:
            raise SystemExit(
                "EVAL_EMBEDDING_API_KEY or OPENAI_API_KEY is required for openai embeddings."
            )
        return OpenAIEmbedder(api_key=api_key, model=model, base_url=base_url)
    if name in {"sentence-transformers", "st", "local", "auto"}:
        return SentenceTransformerEmbedder(model=model or DEFAULT_ST_EMBEDDING_MODEL)
    raise SystemExit(
        f"Unknown embedding provider {provider!r}. "
        "Use sentence-transformers (default), openai, or hash (CI smoke only)."
    )
