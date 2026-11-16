from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARKS_ROOT / "data"
RUNS_DIR = BENCHMARKS_ROOT / "runs"
DEFAULT_BEAM_DATA_DIR = DATA_DIR / "beam"
DEFAULT_HF_DATASET = "Mohammadta/BEAM"
DEFAULT_HF_DATASET_10M = "Mohammadta/BEAM-10M"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_AZURE_JUDGE_MODEL = "gpt-4o"
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"

CHAT_SIZES = ("100K", "500K", "1M", "10M")
HF_DATASET_BY_SIZE = {
    "100K": DEFAULT_HF_DATASET,
    "500K": DEFAULT_HF_DATASET,
    "1M": DEFAULT_HF_DATASET,
    "10M": DEFAULT_HF_DATASET_10M,
}
ABILITY_NAMES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})

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


def model_family(model: str) -> str:
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    for prefix, family in _MODEL_FAMILY_PREFIXES.items():
        if name.startswith(prefix):
            return family
    return name.split("-", 1)[0] or "unknown"


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
    beam_data_dir: Path
    hf_dataset: str
    hf_dataset_10m: str
    runs_dir: Path
    bench_profile: str = "product"
    sc_samples: int = 1
    sc_temperature: float = 0.0
    gap_fill: bool = False
    answer_azure_endpoint: str | None = None
    answer_azure_api_version: str = DEFAULT_AZURE_API_VERSION

    def hf_dataset_for(self, size: str) -> str:
        if size == "10M":
            return self.hf_dataset_10m
        return self.hf_dataset

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            env_file = BENCHMARKS_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        from beam.sota import profile_defaults

        profile = (os.getenv("BENCH_PROFILE") or "product").strip().lower()
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

        judge_model = os.getenv("BENCH_JUDGE_MODEL", "")
        judge_base_url = os.getenv("BENCH_JUDGE_BASE_URL", "")
        judge_api_key = os.getenv("BENCH_JUDGE_API_KEY", "")
        use_azure_judge = azure_available and not judge_model and not judge_base_url
        if use_azure_judge:
            judge_model = os.getenv(
                "AZURE_LARGE_LLM_MODEL", DEFAULT_AZURE_JUDGE_MODEL
            )

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
            beam_data_dir=Path(
                os.getenv("BENCH_BEAM_DATA_DIR", str(DEFAULT_BEAM_DATA_DIR))
            ),
            hf_dataset=os.getenv("BENCH_BEAM_HF_DATASET", DEFAULT_HF_DATASET),
            hf_dataset_10m=os.getenv(
                "BENCH_BEAM_HF_DATASET_10M", DEFAULT_HF_DATASET_10M
            ),
            runs_dir=Path(os.getenv("BENCH_RUNS_DIR", str(RUNS_DIR))),
            bench_profile=str(defaults["bench_profile"]),
            sc_samples=max(1, int(defaults["sc_samples"])),
            sc_temperature=float(defaults["sc_temperature"]),
            gap_fill=bool(defaults["gap_fill"]),
            answer_azure_endpoint=None,
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
