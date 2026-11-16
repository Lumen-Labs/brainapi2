from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARKS_ROOT / "data"
RUNS_DIR = BENCHMARKS_ROOT / "runs"

HF_CLEANED_BASE = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
)

VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "oracle": "longmemeval_oracle.json",
    "m": "longmemeval_m_cleaned.json",
}

DEFAULT_VARIANT = "s"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_SOTA_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_AZURE_JUDGE_MODEL = "gpt-4o"
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_BENCH_PROFILE = "product"

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)

_MODEL_FAMILY_PREFIXES = {
    "deepseek": "deepseek",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "chatgpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "grok": "xai",
    "qwen": "qwen",
    "llama": "meta",
    "mistral": "mistral",
}

TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})


def model_family(model: str) -> str:
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    for prefix, family in _MODEL_FAMILY_PREFIXES.items():
        if name.startswith(prefix):
            return family
    return name.split("-", 1)[0] or "unknown"


def variant_url(variant: str) -> str:
    key = (variant or DEFAULT_VARIANT).strip().lower()
    if key not in VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANT_FILES)}")
    return f"{HF_CLEANED_BASE}/{VARIANT_FILES[key]}"


def variant_path(variant: str) -> Path:
    key = (variant or DEFAULT_VARIANT).strip().lower()
    if key not in VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANT_FILES)}")
    return DATA_DIR / VARIANT_FILES[key]


def profile_defaults(profile: str) -> dict:
    name = (profile or "product").strip().lower()
    if name == "sota":
        return {
            "bench_profile": "sota",
            "sc_samples": int(os.getenv("BENCH_SC_SAMPLES", "5")),
            "sc_temperature": float(os.getenv("BENCH_SC_TEMPERATURE", "0.7")),
            "gap_fill": os.getenv("BENCH_GAP_FILL", "1") not in {"0", "false", "False"},
        }
    return {
        "bench_profile": "product",
        "sc_samples": int(os.getenv("BENCH_SC_SAMPLES", "1")),
        "sc_temperature": 0.0,
        "gap_fill": os.getenv("BENCH_GAP_FILL", "0") in {"1", "true", "True"},
    }


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
    dataset_path: Path
    dataset_url: str
    variant: str
    runs_dir: Path
    bench_profile: str
    sc_samples: int
    sc_temperature: float
    gap_fill: bool
    answer_azure_endpoint: str | None = None
    answer_azure_api_version: str = DEFAULT_AZURE_API_VERSION

    @classmethod
    def load(cls, env_file: Path | None = None, *, variant: str | None = None) -> "Settings":
        if env_file is None:
            env_file = BENCHMARKS_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        profile = (
            os.getenv("BENCH_PROFILE", DEFAULT_BENCH_PROFILE) or DEFAULT_BENCH_PROFILE
        ).strip().lower()
        if profile not in {"product", "sota"}:
            profile = DEFAULT_BENCH_PROFILE
        defaults = profile_defaults(profile)

        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        llm_base_url = os.getenv("BENCH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL) or None
        answer_model = os.getenv("BENCH_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)

        azure_endpoint = (
            os.getenv("BENCH_JUDGE_AZURE_ENDPOINT")
            or os.getenv("AZURE_LARGE_LLM_ENDPOINT")
            or ""
        ).rstrip("/")
        azure_key = (
            os.getenv("BENCH_JUDGE_AZURE_KEY")
            or os.getenv("AZURE_LARGE_LLM_SUBSCRIPTION_KEY")
            or ""
        )
        azure_available = bool(azure_endpoint and azure_key)

        answer_azure_endpoint: str | None = None
        if profile == "sota":
            sota_answer = os.getenv("BENCH_SOTA_ANSWER_MODEL", "").strip()
            if sota_answer:
                answer_model = sota_answer
                sota_base = os.getenv("BENCH_SOTA_LLM_BASE_URL", "").strip()
                if sota_base:
                    llm_base_url = sota_base
            else:
                answer_model = os.getenv(
                    "BENCH_ANSWER_MODEL", DEFAULT_SOTA_ANSWER_MODEL
                )

        judge_model = os.getenv("BENCH_JUDGE_MODEL", "")
        judge_base_url = os.getenv("BENCH_JUDGE_BASE_URL", "")
        judge_api_key = os.getenv("BENCH_JUDGE_API_KEY", "")
        use_azure_judge = azure_available and not judge_model and not judge_base_url
        if use_azure_judge:
            judge_model = os.getenv(
                "AZURE_LARGE_LLM_MODEL", DEFAULT_AZURE_JUDGE_MODEL
            )
        if profile == "sota":
            sota_judge = os.getenv("BENCH_SOTA_JUDGE_MODEL", "").strip()
            if sota_judge:
                judge_model = sota_judge
                use_azure_judge = False
            elif not os.getenv("BENCH_JUDGE_MODEL"):
                judge_model = answer_model
                use_azure_judge = False
                judge_base_url = llm_base_url or ""
                judge_api_key = api_key

        resolved_variant = (
            variant
            or os.getenv("BENCH_LME_VARIANT", DEFAULT_VARIANT)
            or DEFAULT_VARIANT
        ).strip().lower()
        if resolved_variant not in VARIANT_FILES:
            resolved_variant = DEFAULT_VARIANT
        default_path = variant_path(resolved_variant)
        default_url = variant_url(resolved_variant)

        return cls(
            brainapi_url=os.getenv("BRAINAPI_URL", "http://localhost:8000").rstrip("/"),
            brainpat_token=os.getenv("BRAINPAT_TOKEN", ""),
            openai_api_key=api_key,
            llm_base_url=llm_base_url,
            answer_model=answer_model,
            judge_model=judge_model or answer_model,
            judge_api_key=(
                azure_key if use_azure_judge else (judge_api_key or api_key)
            ),
            judge_base_url=(
                None if use_azure_judge else (judge_base_url or llm_base_url)
            ),
            judge_azure_endpoint=azure_endpoint if use_azure_judge else None,
            judge_azure_api_version=os.getenv(
                "BENCH_JUDGE_AZURE_API_VERSION",
                os.getenv("AZURE_LARGE_LLM_API_VERSION", DEFAULT_AZURE_API_VERSION),
            ),
            dataset_path=Path(
                os.getenv("BENCH_LME_DATASET_PATH", str(default_path))
            ),
            dataset_url=os.getenv("BENCH_LME_DATASET_URL", default_url),
            variant=resolved_variant,
            runs_dir=Path(os.getenv("BENCH_RUNS_DIR", str(RUNS_DIR))),
            bench_profile=str(defaults["bench_profile"]),
            sc_samples=max(1, int(defaults["sc_samples"])),
            sc_temperature=float(defaults["sc_temperature"]),
            gap_fill=bool(defaults["gap_fill"]),
            answer_azure_endpoint=answer_azure_endpoint,
            answer_azure_api_version=os.getenv(
                "BENCH_JUDGE_AZURE_API_VERSION",
                os.getenv("AZURE_LARGE_LLM_API_VERSION", DEFAULT_AZURE_API_VERSION),
            ),
        )

    @property
    def judge_shares_answer_family(self) -> bool:
        return model_family(self.judge_model) == model_family(self.answer_model)

    def judge_provider(self) -> str:
        if self.judge_azure_endpoint:
            return "azure"
        return self.judge_base_url or "openai-default"

    def require_brainapi(self) -> None:
        if not self.brainpat_token:
            raise SystemExit(
                "BRAINPAT_TOKEN is required. Copy .env.example to .env and set it."
            )

    def require_llm(self) -> None:
        if not self.openai_api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY (or OPENAI_API_KEY) is required for answer/judge "
                "LLM calls. Copy .env.example to .env and set it."
            )

    def require_judge_llm(self) -> None:
        if not self.judge_api_key:
            raise SystemExit(
                "No judge API key. Set BENCH_JUDGE_API_KEY (with BENCH_JUDGE_BASE_URL "
                "and BENCH_JUDGE_MODEL), or DEEPSEEK_API_KEY / OPENAI_API_KEY to reuse "
                "the answerer's provider."
            )
