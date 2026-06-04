"""
File: /config.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday March 29th 2026 3:27:28 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

# pylint: disable=too-few-public-methods

import logging
import os
from typing import Literal
import dotenv
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_env_name = os.getenv("ENV")
_env_path = _project_root / f".env{'.' + _env_name if _env_name else ''}"
dotenv.load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    normalized = raw.strip().rstrip("/")
    if normalized in ("https://api.openai.com/v1", "https://api.openai.com"):
        return None
    return normalized


class AzureConfig:
    """
    Configuration class for the Azure configuration.
    """

    def __init__(self):
        self.small_llm_model = os.getenv("AZURE_SMALL_LLM_MODEL", "gpt-4o-mini")
        self.large_llm_model = os.getenv("AZURE_LARGE_LLM_MODEL")
        self.llm_api_version = os.getenv("AZURE_LLM_API_VERSION") or os.getenv(
            "AZURE_LARGE_LLM_API_VERSION"
        )
        self.llm_endpoint = os.getenv("AZURE_LLM_ENDPOINT") or os.getenv(
            "AZURE_LARGE_LLM_ENDPOINT"
        )
        self.llm_subscription_key = os.getenv("AZURE_LLM_SUBSCRIPTION_KEY") or os.getenv(
            "AZURE_LARGE_LLM_SUBSCRIPTION_KEY"
        )
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_full_endpoint = os.getenv("AZURE_EMBEDDING_FULL_ENDPOINT")
        self.embedding_key = os.getenv("AZURE_EMBEDDING_KEY")

    def validate_llm(self):
        if [
            self.small_llm_model,
            self.large_llm_model,
            self.llm_api_version,
            self.llm_endpoint,
            self.llm_subscription_key,
        ].count(None) > 0:
            raise ValueError("Azure LLM configuration is not complete")

    def validate_embeddings(self):
        if [self.embedding_full_endpoint, self.embedding_key].count(None) > 0:
            raise ValueError("Azure embeddings configuration is not complete")


class OpenAIConfig:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
        self.small_llm_model = os.getenv("OPENAI_SMALL_LLM_MODEL", "gpt-4o-mini")
        self.large_llm_model = os.getenv("OPENAI_LARGE_LLM_MODEL", "gpt-4o")
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        )

    def validate_llm(self, require_large: bool):
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")
        if not self.small_llm_model:
            raise ValueError("OpenAI small LLM model is not set")
        if require_large and not self.large_llm_model:
            raise ValueError("OpenAI large LLM model is not set")

    def validate_embeddings(self):
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")
        if not self.embedding_model:
            raise ValueError("OpenAI embedding model is not set")


class AnthropicConfig:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.small_llm_model = os.getenv(
            "ANTHROPIC_SMALL_LLM_MODEL", "claude-3-5-haiku-latest"
        )
        self.large_llm_model = os.getenv(
            "ANTHROPIC_LARGE_LLM_MODEL", "claude-sonnet-4-20250514"
        )

    def validate_llm(self, require_large: bool):
        if not self.api_key:
            raise ValueError("Anthropic API key is not set")
        if not self.small_llm_model:
            raise ValueError("Anthropic small LLM model is not set")
        if require_large and not self.large_llm_model:
            raise ValueError("Anthropic large LLM model is not set")


class GCPConfig:
    """
    Configuration class for the GCP configuration.
    """

    def __init__(self):
        credentials_path = os.getenv("GCP_CREDENTIALS_PATH")
        if not credentials_path:
            credentials_path = str(
                Path(__file__).parent.parent / "gcp_credentials.json"
            )
        else:
            credentials_path = os.path.expanduser(credentials_path)
            if not os.path.isabs(credentials_path):
                credentials_path = str(Path(__file__).parent.parent / credentials_path)

        logger.debug("GCP credentials_path: %s", credentials_path)
        self.credentials_path = credentials_path
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.small_llm_model = os.getenv("GCP_SMALL_LLM_MODEL")
        self.large_llm_model = os.getenv("GCP_LARGE_LLM_MODEL")
        self.embedding_model = os.getenv("GCP_EMBEDDING_MODEL", "text-embedding-005")

    def validate_llm(self, require_large: bool):
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {self.credentials_path}. "
                "Please set GCP_CREDENTIALS_PATH environment variable or place gcp_credentials.json in the project root."
            )
        required = [self.credentials_path, self.project_id, self.small_llm_model]
        if require_large:
            required.append(self.large_llm_model)
        if required.count(None) > 0:
            raise ValueError("GCP LLM configuration is not complete")

    def validate_embeddings(self):
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {self.credentials_path}. "
                "Please set GCP_CREDENTIALS_PATH environment variable or place gcp_credentials.json in the project root."
            )
        if [self.credentials_path, self.project_id, self.embedding_model].count(None) > 0:
            raise ValueError("GCP embeddings configuration is not complete")


class BedrockConfig:
    def __init__(self):
        self.region = os.getenv("BEDROCK_REGION")
        self.access_key_id = os.getenv("BEDROCK_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("BEDROCK_SECRET_ACCESS_KEY")
        self.session_token = os.getenv("BEDROCK_SESSION_TOKEN")
        self.small_llm_model = os.getenv("BEDROCK_SMALL_LLM_MODEL")
        self.large_llm_model = os.getenv("BEDROCK_LARGE_LLM_MODEL")
        self.embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL")

    def _validate_auth(self):
        if [self.region, self.access_key_id, self.secret_access_key].count(None) > 0:
            raise ValueError("Bedrock authentication is not complete")

    def validate_llm(self, require_large: bool):
        self._validate_auth()
        required = [self.small_llm_model]
        if require_large:
            required.append(self.large_llm_model)
        if required.count(None) > 0:
            raise ValueError("Bedrock LLM configuration is not complete")

    def validate_embeddings(self):
        self._validate_auth()
        if [self.embedding_model].count(None) > 0:
            raise ValueError("Bedrock embeddings configuration is not complete")


class RedisConfig:
    """
    Configuration class for the Redis configuration.
    """

    def __init__(self):
        self.host = os.getenv("REDIS_HOST")
        port_str = os.getenv("REDIS_PORT")
        self.port = int(port_str) if port_str else None

        if [self.host, self.port].count(None) > 0:
            raise ValueError("Redis configuration is not complete")


class Neo4jConfig:
    """
    Configuration class for the Neo4j configuration.
    """

    def __init__(self):
        self.host = os.getenv("NEO4J_HOST")
        self.port = os.getenv("NEO4J_PORT")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if [self.host, self.port, self.username, self.password].count(None) > 0:
            raise ValueError("Neo4j configuration is not complete")


class PostgreSQLConfig:
    """
    Configuration class for the PostgreSQL graph / data / vector stores.

    BrainAPI uses one Postgres database per brain. Two cluster-level names are
    therefore needed beyond the connection credentials:

      * ``system_database`` — the registry database (mirrors Mongo's
        ``MONGO_SYSTEM_DATABASE``). It stores the ``data_brains`` table that
        catalogs every brain alongside its PAT and metadata.
      * ``maintenance_database`` — the database used solely for
        ``CREATE DATABASE`` statements when a new brain is provisioned. Almost
        every cluster ships with ``postgres`` available for this role.

    ``POSTGRES_DATABASE`` is retained as the source of truth for the system
    database to preserve backwards compatibility with existing deployments.
    ``POSTGRES_SYSTEM_DATABASE`` takes precedence when set explicitly.
    """

    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST")
        port_str = os.getenv("POSTGRES_PORT")
        self.port = int(port_str) if port_str else None
        self.username = os.getenv("POSTGRES_USERNAME")
        self.password = os.getenv("POSTGRES_PASSWORD")

        system_db = os.getenv("POSTGRES_SYSTEM_DATABASE") or os.getenv(
            "POSTGRES_DATABASE"
        )
        self.system_database = system_db
        self.database = system_db
        self.maintenance_database = os.getenv(
            "POSTGRES_MAINTENANCE_DATABASE", "postgres"
        )

    def validate_credentials(self) -> None:
        """Validate only what is required to open a Postgres connection."""
        if [self.host, self.port, self.username, self.password].count(None) > 0:
            raise ValueError("PostgreSQL connection configuration is not complete")

    def validate(self) -> None:
        self.validate_credentials()
        if not self.system_database:
            raise ValueError(
                "PostgreSQL system database is not configured "
                "(set POSTGRES_DATABASE or POSTGRES_SYSTEM_DATABASE)"
            )


class MilvusConfig:
    """
    Configuration class for the Milvus configuration.
    """

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST")
        port_str = os.getenv("MILVUS_PORT")
        self.port = int(port_str) if port_str else None
        self.uri = os.getenv("MILVUS_URI")
        self.token = os.getenv("MILVUS_TOKEN")
        if [self.host, self.port].count(None) > 0 and [self.uri, self.token].count(
            None
        ) > 0:
            raise ValueError("Milvus configuration is not complete")


class EmbeddingsConfig:
    """
    Configuration class for the Embeddings configuration.
    """

    def __init__(self, mode: str):
        self.local_model = os.getenv("EMBEDDINGS_LOCAL_MODEL")
        self.small_model = os.getenv("EMBEDDINGS_SMALL_MODEL")

        self.embedding_nodes_dimension = (
            int(os.getenv("EMBEDDING_NODES_DIMENSION"))
            if os.getenv("EMBEDDING_NODES_DIMENSION")
            else None
        )
        self.embedding_triplets_dimension = os.getenv("EMBEDDING_TRIPLETS_DIMENSION")
        self.embedding_observations_dimension = (
            int(os.getenv("EMBEDDING_OBSERVATIONS_DIMENSION"))
            if os.getenv("EMBEDDING_OBSERVATIONS_DIMENSION")
            else None
        )
        self.embedding_data_dimension = (
            int(os.getenv("EMBEDDING_DATA_DIMENSION"))
            if os.getenv("EMBEDDING_DATA_DIMENSION")
            else None
        )
        self.embedding_relationships_dimension = (
            int(os.getenv("EMBEDDING_RELATIONSHIPS_DIMENSION"))
            if os.getenv("EMBEDDING_RELATIONSHIPS_DIMENSION")
            else None
        )

        if [
            self.local_model,
            self.small_model,
            self.embedding_nodes_dimension,
            self.embedding_triplets_dimension,
            self.embedding_observations_dimension,
            self.embedding_data_dimension,
            self.embedding_relationships_dimension,
        ].count(None) > 0:
            raise ValueError("Embeddings configuration is not complete")


class MongoConfig:
    """
    Configuration class for the Mongo configuration.
    """

    def __init__(self):
        self.host = os.getenv("MONGO_HOST")
        self.port = int(os.getenv("MONGO_PORT")) if os.getenv("MONGO_PORT") else None
        self.username = os.getenv("MONGO_USERNAME")
        self.password = os.getenv("MONGO_PASSWORD")

        self.connection_string = os.getenv("MONGO_CONNECTION_STRING")
        self.system_database = os.getenv("MONGO_SYSTEM_DATABASE", "system")

        if [self.host, self.port, self.username, self.password].count(
            None
        ) > 0 and not self.connection_string:
            raise ValueError("Mongo configuration is not complete")


class CeleryConfig:
    """
    Configuration class for the Celery configuration.
    """

    def __init__(self):
        """
        Initialize CeleryConfig by loading the worker concurrency setting from the environment.

        Sets `self.worker_concurrency` from the `CELERY_WORKER_CONCURRENCY` environment variable and validates its presence.

        Raises:
            ValueError: If `CELERY_WORKER_CONCURRENCY` is not set.
        """
        self.worker_concurrency = os.getenv("CELERY_WORKER_CONCURRENCY")
        if [self.worker_concurrency].count(None) > 0:
            raise ValueError("Celery configuration is not complete")


class OllamaConfig:
    """
    Configuration class for the Ollama configuration.
    """

    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST")
        self.port = os.getenv("OLLAMA_PORT")
        self.llm_small_model = os.getenv("OLLAMA_LLM_SMALL_MODEL")
        self.llm_large_model = os.getenv("OLLAMA_LLM_LARGE_MODEL")
        if [self.host, self.port, self.llm_small_model, self.llm_large_model].count(
            None
        ) > 0:
            raise ValueError("Ollama configuration is not complete")


class PricingConfig:
    """
    Configuration class for the Pricing configuration.
    """

    def __init__(self):
        """
        Initialize pricing configuration from environment variables.

        Attributes:
            input_token_price (float): Price per input token from INPUT_TOKEN_PRICE, defaults to 0.0 if unset.
            output_token_price (float): Price per output token from OUTPUT_TOKEN_PRICE, defaults to 0.0 if unset.
        """
        self.input_token_price = float(os.getenv("INPUT_TOKEN_PRICE", 0))
        self.output_token_price = float(os.getenv("OUTPUT_TOKEN_PRICE", 0))


class SpacyConfig:
    """
    Configuration class for the Spacy configuration.
    """

    def __init__(self):
        self.keep_models_in_memory = (
            os.getenv("SPACY_KEEP_MODELS_IN_MEMORY", "false") == "true"
        )


_MODES = ("local", "remote")
_PROVIDERS = ("ollama", "azure", "openai", "anthropic", "gcp_vertex", "amazon_bedrock")


class Config:
    """
    Configuration class for the application.
    """

    def __init__(self):
        """
        Initialize the application's central configuration by composing environment-backed sub-configurations and loading runtime flags.

        This constructor instantiates Azure, Redis, Neo4j, Milvus, Embeddings, Mongo, GCP, Celery, and Pricing configuration objects, reads the RUN_GRAPH_CONSOLIDATOR flag into `run_graph_consolidator`, and loads the `BRAINPAT_TOKEN` into `brainpat_token`.

        Raises:
            ValueError: If `BRAINPAT_TOKEN` is not set in the environment.
        """
        self.brainpat_token = os.getenv("BRAINPAT_TOKEN")
        if not self.brainpat_token:
            raise ValueError("BrainPAT token is not set")

        self.models_mode = os.getenv("MODELS_MODE")
        if self.models_mode not in _MODES:
            raise ValueError(f"Invalid MODELS_MODE: {self.models_mode}")

        default_small_provider = "ollama" if self.models_mode == "local" else "gcp_vertex"
        default_large_provider = "ollama" if self.models_mode == "local" else "azure"
        default_embeddings_provider = (
            "ollama" if self.models_mode == "local" else "azure"
        )

        self.llm_small_provider = os.getenv("LLM_SMALL_PROVIDER", default_small_provider)
        self.llm_large_provider = os.getenv("LLM_LARGE_PROVIDER", default_large_provider)
        self.embeddings_provider = os.getenv(
            "EMBEDDINGS_PROVIDER", default_embeddings_provider
        )

        for provider in (
            self.llm_small_provider,
            self.llm_large_provider,
            self.embeddings_provider,
        ):
            if provider not in _PROVIDERS:
                raise ValueError(f"Invalid provider: {provider}")

        use_azure_llm = self.llm_small_provider == "azure" or self.llm_large_provider == "azure"
        use_azure_embeddings = self.embeddings_provider == "azure"
        use_gcp_llm = self.llm_small_provider == "gcp_vertex" or self.llm_large_provider == "gcp_vertex"
        use_gcp_large = self.llm_large_provider == "gcp_vertex"
        use_gcp_embeddings = self.embeddings_provider == "gcp_vertex"
        use_bedrock_llm = (
            self.llm_small_provider == "amazon_bedrock"
            or self.llm_large_provider == "amazon_bedrock"
        )
        use_bedrock_large = self.llm_large_provider == "amazon_bedrock"
        use_bedrock_embeddings = self.embeddings_provider == "amazon_bedrock"
        use_openai_llm = (
            self.llm_small_provider == "openai" or self.llm_large_provider == "openai"
        )
        use_openai_large = self.llm_large_provider == "openai"
        use_openai_embeddings = self.embeddings_provider == "openai"
        use_anthropic_llm = (
            self.llm_small_provider == "anthropic"
            or self.llm_large_provider == "anthropic"
        )
        use_anthropic_large = self.llm_large_provider == "anthropic"
        use_ollama = (
            self.llm_small_provider == "ollama"
            or self.llm_large_provider == "ollama"
            or self.embeddings_provider == "ollama"
        )

        self.azure = AzureConfig() if use_azure_llm or use_azure_embeddings else None
        if self.azure is not None:
            if use_azure_llm:
                self.azure.validate_llm()
            if use_azure_embeddings:
                self.azure.validate_embeddings()

        self.gcp = GCPConfig() if use_gcp_llm or use_gcp_embeddings else None
        if self.gcp is not None:
            if use_gcp_llm:
                self.gcp.validate_llm(require_large=use_gcp_large)
            if use_gcp_embeddings:
                self.gcp.validate_embeddings()

        self.bedrock = BedrockConfig() if use_bedrock_llm or use_bedrock_embeddings else None
        if self.bedrock is not None:
            if use_bedrock_llm:
                self.bedrock.validate_llm(require_large=use_bedrock_large)
            if use_bedrock_embeddings:
                self.bedrock.validate_embeddings()

        self.openai = OpenAIConfig() if use_openai_llm or use_openai_embeddings else None
        if self.openai is not None:
            if use_openai_llm:
                self.openai.validate_llm(require_large=use_openai_large)
            if use_openai_embeddings:
                self.openai.validate_embeddings()

        self.anthropic = AnthropicConfig() if use_anthropic_llm else None
        if self.anthropic is not None:
            self.anthropic.validate_llm(require_large=use_anthropic_large)

        self.ollama = OllamaConfig() if use_ollama else None
        self.embeddings = EmbeddingsConfig(mode=self.models_mode)

        self.vector_db = os.getenv("VECTOR_DB", "milvus")
        self.data_db = os.getenv("DATA_DB", "mongo")
        self.graph_db = os.getenv("GRAPH_DB", "neo4j")

        self.redis = RedisConfig()
        self.neo4j = Neo4jConfig()
        self.postgresql = PostgreSQLConfig()
        self.milvus = MilvusConfig()
        self.mongo = MongoConfig()
        self.celery = CeleryConfig()
        self.pricing = PricingConfig()
        self.spacy = SpacyConfig()

        self.run_graph_consolidator = (
            os.getenv("RUN_GRAPH_CONSOLIDATOR", "true") == "true"
        )
        self.docparser_endpoint = os.getenv("DOCPARSER_ENDPOINT")
        self.docparser_token = os.getenv("DOCPARSER_TOKEN")
        self.app_host = os.getenv("APP_HOST")
        self.pipeline_mode: Literal["lightweight", "accurate"] = os.getenv(
            "PIPELINE_MODE"
        )
        self.ocr_mode: Literal["docling", "docparser"] = os.getenv(
            "OCR_MODE", "docling"
        )
        self.agentic_architecture: Literal["custom", "langchain"] = os.getenv(
            "AGENTIC_ARCHITECTURE", "custom"
        )


config = Config()
