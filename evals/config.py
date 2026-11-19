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
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_ST_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
CHEAP_ST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHEAP_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_EN_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SOTA_BASELINE_PROFILES = frozenset({"sota", "strong", "bge", "default", ""})
CHEAP_BASELINE_PROFILES = frozenset({"minilm", "cheap", "light"})
DEFAULT_BRAIN_PREFIX = "evaluc"
DEFAULT_CONFLICT_LIST_PATH = "/retrieve/conflicts"
DEFAULT_CONFLICT_SCAN_PATH = "/retrieve/conflicts/scan"
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


def resolve_baseline_profile(value: str | None = None) -> str:
    raw = (value if value is not None else _first_env("EVAL_BASELINE_PROFILE")).strip().lower()
    if raw in CHEAP_BASELINE_PROFILES:
        return "minilm"
    if raw in SOTA_BASELINE_PROFILES or not raw:
        return "sota"
    raise SystemExit(
        f"EVAL_BASELINE_PROFILE={value!r} is not supported. Use 'sota' "
        f"(BAAI/bge-large-en-v1.5 + BAAI/bge-reranker-v2-m3) or 'minilm' "
        f"({CHEAP_ST_EMBEDDING_MODEL} + {CHEAP_CROSS_ENCODER})."
    )


def models_for_baseline_profile(profile: str) -> tuple[str, str]:
    if profile == "minilm":
        return CHEAP_ST_EMBEDDING_MODEL, CHEAP_CROSS_ENCODER
    return DEFAULT_ST_EMBEDDING_MODEL, DEFAULT_CROSS_ENCODER


def query_prefix_for_embedding_model(model: str) -> str:
    name = (model or "").strip().lower()
    if not name:
        return ""
    if "bge-m3" in name:
        return ""
    if "bge-large-en" in name or "bge-base-en" in name or "bge-small-en" in name:
        return BGE_EN_QUERY_PREFIX
    return ""


def embedding_dim_for(model: str, *, default: int = 1024) -> int:
    name = (model or "").strip().lower()
    if "minilm" in name:
        return 384
    if "bge-small" in name:
        return 384
    if "bge-base" in name and "m3" not in name:
        return 768
    if "bge-large" in name or "bge-m3" in name:
        return 1024
    return default


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _optional_path(*names: str, default: str = "") -> str | None:
    raw = _first_env(*names, default=default).strip()
    if not raw or raw.lower() in {"off", "none", "disabled"}:
        return None
    return raw


def resolve_llm_model() -> str:
    seen: list[tuple[str, str]] = []
    for name in LLM_MODEL_ENV_KEYS:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        seen.append((name, str(raw).strip()))
    unique = {value for _, value in seen}
    if len(unique) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in seen)
        raise SystemExit(
            "Conflicting LLM model settings: "
            f"{detail}. Use EVAL_LLM_MODEL for the shared answerer and judge."
        )
    if unique:
        return unique.pop()
    return ""


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
    conflict_scan_path: str | None
    conflict_plugin: str
    runs_dir: Path
    datasets_dir: Path
    scale: str
    brain_prefix: str
    baseline_profile: str = "sota"
    llm_provider: str = ""
    expected_brainapi_model: str = ""
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

        llm_model = resolve_llm_model()
        llm_provider = _first_env("EVAL_LLM_PROVIDER")
        llm_base_url = _first_env("EVAL_LLM_BASE_URL") or None
        judge_base = _first_env("EVAL_JUDGE_BASE_URL", "BENCH_JUDGE_BASE_URL")
        if llm_base_url and judge_base and judge_base.rstrip("/") != llm_base_url.rstrip("/"):
            raise SystemExit(
                "EVAL_JUDGE_BASE_URL must match EVAL_LLM_BASE_URL. The harness "
                "uses one shared chat client for answering and judging."
            )

        embedding_key = _first_env(
            "EVAL_EMBEDDING_API_KEY",
            "OPENAI_API_KEY",
            "AZURE_EMBEDDING_KEY",
        )
        expected_brainapi_model = _first_env("EVAL_EXPECTED_BRAINAPI_MODEL")
        baseline_profile = resolve_baseline_profile()
        default_st, default_ce = models_for_baseline_profile(baseline_profile)
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
            llm_provider=llm_provider,
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
                    else default_st
                ),
            ),
            reranker=_first_env("EVAL_RERANKER", default="cross-encoder").lower(),
            cross_encoder_model=_first_env(
                "EVAL_CROSS_ENCODER_MODEL", default=default_ce
            ),
            as_of_field=_first_env("EVAL_AS_OF_FIELD") or None,
            conflict_path=_optional_path(
                "EVAL_CONFLICT_PATH", default=DEFAULT_CONFLICT_LIST_PATH
            ),
            conflict_scan_path=_optional_path(
                "EVAL_CONFLICT_SCAN_PATH", default=DEFAULT_CONFLICT_SCAN_PATH
            ),
            conflict_plugin=_first_env(
                "EVAL_CONFLICT_PLUGIN", default="temporal_conflicts"
            ),
            runs_dir=Path(_first_env("EVAL_RUNS_DIR", default=str(RUNS_DIR))),
            datasets_dir=Path(_first_env("EVAL_DATASETS_DIR", default=str(DATASETS_DIR))),
            scale=_first_env("EVAL_SCALE", default=DEFAULT_SCALE),
            brain_prefix=_first_env(
                "EVAL_BRAIN_PREFIX", default=DEFAULT_BRAIN_PREFIX
            ),
            baseline_profile=baseline_profile,
            expected_brainapi_model=expected_brainapi_model,
        )

    def require_brainapi(self) -> None:
        if not self.brainpat_token:
            raise SystemExit(
                "BRAINPAT_TOKEN (or EVAL_API_KEY) is required to talk to BrainAPI. "
                "Copy evals/.env.example to evals/.env and set it. "
                "Do not commit secrets."
            )

    def require_llm(self) -> None:
        if not (self.llm_model or "").strip():
            raise SystemExit(
                "EVAL_LLM_MODEL is required for LLM answering and judging. "
                "Local smoke can run with --answerer extractive (no LLM)."
            )
        if not self.llm_api_key:
            raise SystemExit(
                "EVAL_LLM_API_KEY is required for LLM answering and judging. "
                "Local smoke can run with --answerer extractive (no LLM)."
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
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url or "",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "reranker": self.reranker,
            "reranker_model": self.cross_encoder_model,
            "baseline_profile": self.baseline_profile,
            "scale": self.scale,
            "as_of_field": self.as_of_field or "",
            "conflict_path": self.conflict_path or "",
            "conflict_scan_path": self.conflict_scan_path or "",
            "llm_key_present": "yes" if self.llm_api_key else "no",
            "embedding_key_present": "yes" if self.embedding_api_key else "no",
            "expected_brainapi_model": self.expected_brainapi_model,
        }
