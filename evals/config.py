from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from evals import SCHEMA_VERSION

EVALS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALS_ROOT.parent
DATASETS_DIR = EVALS_ROOT / "datasets"
RUNS_DIR = EVALS_ROOT / "runs"
DEFAULT_SCALE = "local/small-scale"
DEFAULT_LLM_MODEL = "claude-haiku-5-5"
DEFAULT_LLM_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_ST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BRAIN_PREFIX = "evaluc"
PROMPT_BUDGET = {
    "max_passages": 8,
    "max_chars": 6000,
    "max_answer_tokens": 256,
    "temperature": 0.0,
}
USE_CASES = (
    "versioned_rules",
    "policy_conflicts",
    "expert_knowhow",
    "decision_precedent",
)
REQUIRED_USE_CASES = (
    "versioned_rules",
    "policy_conflicts",
    "expert_knowhow",
)
OPTIONAL_USE_CASES = ("decision_precedent",)
FORBIDDEN_BRAIN_PREFIXES = (
    "beam",
    "demorecsys",
    "lme",
    "locomoconv",
    "longmemeval",
    "searchbench",
)
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})
LLM_MODEL_ENV_KEYS = (
    "EVAL_LLM_MODEL",
    "EVAL_ANSWER_MODEL",
    "EVAL_JUDGE_MODEL",
    "BENCH_ANSWER_MODEL",
    "BENCH_JUDGE_MODEL",
)
ALLOWED_LLM_ALIASES = {
    "claude-haiku-5-5": DEFAULT_LLM_MODEL,
    "claude-haiku-5.5": DEFAULT_LLM_MODEL,
    "anthropic.claude-haiku-5-5": DEFAULT_LLM_MODEL,
    "anthropic/claude-haiku-5-5": DEFAULT_LLM_MODEL,
}
FORBIDDEN_LLM_HOST_MARKERS = (
    "api.deepseek.com",
    "api.openai.com",
    "openai.azure.com",
    "cognitiveservices.azure.com",
)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _llm_model_error(source: str, value: str) -> SystemExit:
    return SystemExit(
        f"{source}={value!r} is not allowed. The eval harness may only use Claude "
        f"Haiku 5.5 (model id {DEFAULT_LLM_MODEL!r}) for the shared answerer, the "
        "LLM judge, and any LLM BrainAPI calls. Embeddings and the cross-encoder "
        "reranker are not LLMs and may stay as configured."
    )


def canonicalize_llm_model(value: str, *, source: str = "EVAL_LLM_MODEL") -> str:
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_LLM_MODEL
    key = raw.lower().replace("_", "-")
    mapped = ALLOWED_LLM_ALIASES.get(key)
    if mapped:
        return mapped
    raise _llm_model_error(source, raw)


def resolve_llm_model() -> str:
    seen: list[str] = []
    for name in LLM_MODEL_ENV_KEYS:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        seen.append(canonicalize_llm_model(raw, source=name))
    unique = set(seen)
    if unique and unique != {DEFAULT_LLM_MODEL}:
        raise SystemExit(
            "Conflicting LLM model settings. The eval harness uses a single model "
            f"id, {DEFAULT_LLM_MODEL!r}."
        )
    return DEFAULT_LLM_MODEL


def assert_llm_endpoint(url: str | None, *, source: str) -> None:
    if not url:
        return
    host = (urlparse(url).hostname or url).lower()
    lowered = url.lower()
    for marker in FORBIDDEN_LLM_HOST_MARKERS:
        if marker in host or marker in lowered:
            raise SystemExit(
                f"{source}={url!r} is a DeepSeek/OpenAI/Azure chat endpoint. "
                f"Eval LLM traffic must use Claude Haiku 5.5 ({DEFAULT_LLM_MODEL}) "
                f"via Anthropic or an Anthropic-compatible gateway."
            )


def validate_brain_id(brain_id: str) -> str:
    bid = (brain_id or "").strip()
    if not bid:
        raise SystemExit("brain_id is required.")
    lowered = bid.lower()
    if any(lowered.startswith(prefix) for prefix in FORBIDDEN_BRAIN_PREFIXES):
        raise SystemExit(
            f"Refusing brain_id={bid!r}. Use a dedicated {DEFAULT_BRAIN_PREFIX}* "
            "brain. Never wipe locomo/beam/search/recsys eval brains."
        )
    if not lowered.startswith(DEFAULT_BRAIN_PREFIX):
        raise SystemExit(
            f"Refusing brain_id={bid!r}. Eval brains must start with "
            f"{DEFAULT_BRAIN_PREFIX!r}."
        )
    return bid


@dataclass
class Settings:
    brainapi_url: str
    brainpat_token: str
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    embedding_provider: str
    embedding_api_key: str
    embedding_base_url: str | None
    embedding_model: str
    reranker: str
    cross_encoder_model: str
    as_of_field: str | None
    conflict_path: str | None
    conflict_plugin: str
    runs_dir: Path
    datasets_dir: Path
    scale: str
    brain_prefix: str
    prompt_budget: dict[str, int | float] = field(default_factory=lambda: dict(PROMPT_BUDGET))
    schema_version: str = SCHEMA_VERSION

    @property
    def answer_model(self) -> str:
        return self.llm_model

    @property
    def judge_model(self) -> str:
        return self.llm_model

    @property
    def openai_api_key(self) -> str:
        return self.llm_api_key

    @property
    def judge_api_key(self) -> str:
        return self.llm_api_key

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        candidates = []
        if env_file is not None:
            candidates.append(env_file)
        candidates.extend(
            [
                EVALS_ROOT / ".env",
                REPO_ROOT / "benchmarks" / ".env",
                REPO_ROOT / ".env",
            ]
        )
        loaded = False
        for path in candidates:
            if path.exists():
                load_dotenv(path, override=False)
                loaded = True
        if not loaded:
            load_dotenv(override=False)

        azure_judge = _first_env(
            "EVAL_JUDGE_AZURE_ENDPOINT", "EVAL_JUDGE_AZURE_KEY"
        )
        if azure_judge:
            raise SystemExit(
                "EVAL_JUDGE_AZURE_* is set. Azure OpenAI judges are not allowed; "
                f"the eval harness uses only Claude Haiku 5.5 ({DEFAULT_LLM_MODEL})."
            )

        llm_model = resolve_llm_model()
        llm_base_url = _first_env(
            "EVAL_LLM_BASE_URL", default=DEFAULT_LLM_BASE_URL
        ) or None
        assert_llm_endpoint(llm_base_url, source="EVAL_LLM_BASE_URL")
        judge_base = _first_env("EVAL_JUDGE_BASE_URL", "BENCH_JUDGE_BASE_URL")
        if judge_base:
            assert_llm_endpoint(judge_base, source="EVAL_JUDGE_BASE_URL")
            if llm_base_url and judge_base.rstrip("/") != llm_base_url.rstrip("/"):
                raise SystemExit(
                    "EVAL_JUDGE_BASE_URL must match EVAL_LLM_BASE_URL. The harness "
                    f"uses one Claude Haiku 5.5 client ({DEFAULT_LLM_MODEL})."
                )

        embedding_key = _first_env(
            "EVAL_EMBEDDING_API_KEY",
            "OPENAI_API_KEY",
            "AZURE_EMBEDDING_KEY",
        )
        return cls(
            brainapi_url=_first_env(
                "EVAL_BRAINAPI_URL", "BRAINAPI_URL", default="http://localhost:8000"
            ).rstrip("/"),
            brainpat_token=_first_env(
                "EVAL_API_KEY", "BRAINPAT_TOKEN", "EVAL_BRAINPAT_TOKEN"
            ),
            llm_api_key=_first_env("EVAL_LLM_API_KEY", "ANTHROPIC_API_KEY"),
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            embedding_provider=_first_env(
                "EVAL_EMBEDDING_PROVIDER", default="sentence-transformers"
            ).lower(),
            embedding_api_key=embedding_key,
            embedding_base_url=_first_env(
                "EVAL_EMBEDDING_BASE_URL", "OPENAI_BASE_URL"
            )
            or None,
            embedding_model=_first_env(
                "EVAL_EMBEDDING_MODEL",
                default=(
                    DEFAULT_EMBEDDING_MODEL
                    if _first_env(
                        "EVAL_EMBEDDING_PROVIDER", default="sentence-transformers"
                    ).lower()
                    in {"openai", "azure"}
                    else DEFAULT_ST_EMBEDDING_MODEL
                ),
            ),
            reranker=_first_env("EVAL_RERANKER", default="cross-encoder").lower(),
            cross_encoder_model=_first_env(
                "EVAL_CROSS_ENCODER_MODEL", default=DEFAULT_CROSS_ENCODER
            ),
            as_of_field=_first_env("EVAL_AS_OF_FIELD") or None,
            conflict_path=_first_env("EVAL_CONFLICT_PATH") or None,
            conflict_plugin=_first_env(
                "EVAL_CONFLICT_PLUGIN", default="temporal_conflicts"
            ),
            runs_dir=Path(_first_env("EVAL_RUNS_DIR", default=str(RUNS_DIR))),
            datasets_dir=Path(_first_env("EVAL_DATASETS_DIR", default=str(DATASETS_DIR))),
            scale=_first_env("EVAL_SCALE", default=DEFAULT_SCALE),
            brain_prefix=_first_env(
                "EVAL_BRAIN_PREFIX", default=DEFAULT_BRAIN_PREFIX
            ),
        )

    def require_brainapi(self) -> None:
        if not self.brainpat_token:
            raise SystemExit(
                "BRAINPAT_TOKEN (or EVAL_API_KEY) is required to talk to BrainAPI. "
                "Copy evals/.env.example to evals/.env and set it. "
                "Do not commit secrets."
            )

    def require_llm(self) -> None:
        canonicalize_llm_model(self.llm_model, source="llm_model")
        if not self.llm_api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY or EVAL_LLM_API_KEY is required for LLM answering "
                f"and judging ({DEFAULT_LLM_MODEL}). Local smoke can run with "
                "--answerer extractive (no LLM)."
            )

    def require_judge_llm(self) -> None:
        self.require_llm()

    def brain_id_for(self, use_case: str) -> str:
        slug = use_case.replace("_", "")
        return validate_brain_id(f"{self.brain_prefix}{slug}")

    def redacted_env(self) -> dict[str, str]:
        def mask(value: str) -> str:
            if not value:
                return ""
            if len(value) <= 8:
                return "***"
            return value[:3] + "…" + value[-2:]

        return {
            "brainapi_url": self.brainapi_url,
            "brainpat_token": mask(self.brainpat_token),
            "llm_model": self.llm_model,
            "answer_model": self.llm_model,
            "judge_model": self.llm_model,
            "llm_base_url": self.llm_base_url or "",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "reranker": self.reranker,
            "scale": self.scale,
            "as_of_field": self.as_of_field or "",
            "conflict_path": self.conflict_path or "",
            "llm_key_present": "yes" if self.llm_api_key else "no",
            "embedding_key_present": "yes" if self.embedding_api_key else "no",
        }
