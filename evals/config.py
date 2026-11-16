from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from evals import SCHEMA_VERSION

EVALS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALS_ROOT.parent
DATASETS_DIR = EVALS_ROOT / "datasets"
RUNS_DIR = EVALS_ROOT / "runs"
DEFAULT_SCALE = "local/small-scale"
DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
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


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


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
    openai_api_key: str
    llm_base_url: str | None
    answer_model: str
    judge_model: str
    judge_api_key: str
    judge_base_url: str | None
    judge_azure_endpoint: str | None
    judge_azure_api_version: str
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

        api_key = _first_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "EVAL_LLM_API_KEY")
        azure_endpoint = _first_env(
            "EVAL_JUDGE_AZURE_ENDPOINT", "AZURE_LARGE_LLM_ENDPOINT"
        ).rstrip("/")
        azure_key = _first_env(
            "EVAL_JUDGE_AZURE_KEY", "AZURE_LARGE_LLM_SUBSCRIPTION_KEY"
        )
        judge_model = _first_env("EVAL_JUDGE_MODEL", "BENCH_JUDGE_MODEL")
        judge_base = _first_env("EVAL_JUDGE_BASE_URL", "BENCH_JUDGE_BASE_URL")
        judge_key = _first_env("EVAL_JUDGE_API_KEY", "BENCH_JUDGE_API_KEY")
        use_azure = bool(azure_endpoint and azure_key and not judge_model and not judge_base)
        if use_azure:
            judge_model = _first_env("AZURE_LARGE_LLM_MODEL", default="gpt-4o")
            judge_key = azure_key

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
            openai_api_key=api_key,
            llm_base_url=_first_env(
                "EVAL_LLM_BASE_URL", "BENCH_LLM_BASE_URL", default=DEFAULT_LLM_BASE_URL
            )
            or None,
            answer_model=_first_env(
                "EVAL_ANSWER_MODEL", "BENCH_ANSWER_MODEL", default=DEFAULT_ANSWER_MODEL
            ),
            judge_model=judge_model or _first_env(
                "EVAL_ANSWER_MODEL", default=DEFAULT_ANSWER_MODEL
            ),
            judge_api_key=judge_key or api_key,
            judge_base_url=None if use_azure else (judge_base or _first_env(
                "EVAL_LLM_BASE_URL", "BENCH_LLM_BASE_URL", default=DEFAULT_LLM_BASE_URL
            ) or None),
            judge_azure_endpoint=azure_endpoint if use_azure else None,
            judge_azure_api_version=_first_env(
                "EVAL_JUDGE_AZURE_API_VERSION",
                "AZURE_LARGE_LLM_API_VERSION",
                default="2024-12-01-preview",
            ),
            embedding_provider=_first_env(
                "EVAL_EMBEDDING_PROVIDER", default="hash"
            ).lower(),
            embedding_api_key=embedding_key,
            embedding_base_url=_first_env(
                "EVAL_EMBEDDING_BASE_URL", "OPENAI_BASE_URL"
            )
            or None,
            embedding_model=_first_env(
                "EVAL_EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL
            ),
            reranker=_first_env("EVAL_RERANKER", default="lexical").lower(),
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
        if not self.openai_api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY or OPENAI_API_KEY is required for LLM answering. "
                "Local smoke can run with --answerer extractive (no LLM)."
            )

    def require_judge_llm(self) -> None:
        if not self.judge_api_key:
            raise SystemExit(
                "No judge API key. Set EVAL_JUDGE_API_KEY or DEEPSEEK_API_KEY / "
                "OPENAI_API_KEY. Exact-match scoring does not need a judge."
            )

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
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "reranker": self.reranker,
            "scale": self.scale,
            "as_of_field": self.as_of_field or "",
            "conflict_path": self.conflict_path or "",
            "llm_key_present": "yes" if self.openai_api_key else "no",
            "embedding_key_present": "yes" if self.embedding_api_key else "no",
        }
