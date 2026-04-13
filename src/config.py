"""
File: /config.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday March 4th 2026 9:35:41 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

# pylint: disable=too-few-public-methods

import logging
import os
from typing import Literal, cast
import dotenv
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_env_name = os.getenv("ENV")
_env_path = _project_root / f".env{'.' + _env_name if _env_name else ''}"
dotenv.load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


class AzureConfig:
    """
    Configuration class for the Azure configuration.
    """

    def __init__(self):
        self.large_llm_model = os.getenv("AZURE_LARGE_LLM_MODEL")
        self.large_llm_api_version = os.getenv("AZURE_LARGE_LLM_API_VERSION")
        self.large_llm_endpoint = os.getenv("AZURE_LARGE_LLM_ENDPOINT")
        self.large_llm_subscription_key = os.getenv("AZURE_LARGE_LLM_SUBSCRIPTION_KEY")
        self.embedding_full_endpoint = os.getenv("AZURE_EMBEDDING_FULL_ENDPOINT")
        self.embedding_key = os.getenv("AZURE_EMBEDDING_KEY")

        if [
            self.large_llm_model,
            self.large_llm_api_version,
            self.large_llm_endpoint,
            self.large_llm_subscription_key,
            self.embedding_full_endpoint,
            self.embedding_key,
        ].count(None) > 0:
            raise ValueError("Azure configuration is not complete")


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

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {credentials_path}. "
                "Please set GCP_CREDENTIALS_PATH environment variable or place gcp_credentials.json in the project root."
            )
        logger.debug("GCP credentials_path: %s", credentials_path)
        self.credentials_path = credentials_path
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.small_llm_model = os.getenv("GCP_SMALL_LLM_MODEL")

        if [self.credentials_path, self.project_id, self.small_llm_model].count(
            None
        ) > 0:
            raise ValueError("GCP configuration is not complete")


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
        self.full_endpoint = os.getenv("EMBEDDINGS_FULL_ENDPOINT")
        self.key = os.getenv("EMBEDDINGS_KEY")
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

        if mode == "local":
            if [
                self.local_model,
                self.small_model,
                self.embedding_nodes_dimension,
                self.embedding_triplets_dimension,
                self.embedding_observations_dimension,
                self.embedding_data_dimension,
                self.embedding_relationships_dimension,
            ].count(None) > 0:
                raise ValueError("Embeddings configuration is not complete (local)")
        else:
            if [
                self.full_endpoint,
                self.key,
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


MODEL_MODES = ("local", "remote")
PIPELINE_MODES = ("lightweight", "accurate")
OCR_MODES = ("docling", "docparser")
AGENTIC_ARCHITECTURES = ("custom", "langchain")


def _read_mode(name: str, allowed_values: tuple[str, ...], default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value not in allowed_values:
        raise ValueError(f"Invalid {name}: {value}")
    return value


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

        self.models_mode = cast(
            Literal["local", "remote"], _read_mode("MODELS_MODE", MODEL_MODES)
        )

        if self.models_mode == "local":
            self.azure = None
            self.gcp = None
            self.embeddings = EmbeddingsConfig(mode="local")
        else:
            self.azure = AzureConfig()
            self.gcp = GCPConfig()
            self.embeddings = EmbeddingsConfig(mode="remote")

        self.redis = RedisConfig()
        self.neo4j = Neo4jConfig()
        self.milvus = MilvusConfig()
        self.mongo = MongoConfig()
        self.celery = CeleryConfig()
        self.pricing = PricingConfig()
        self.spacy = SpacyConfig()
        self.ollama = OllamaConfig()

        self.run_graph_consolidator = (
            os.getenv("RUN_GRAPH_CONSOLIDATOR", "true") == "true"
        )
        self.docparser_endpoint = os.getenv("DOCPARSER_ENDPOINT")
        self.docparser_token = os.getenv("DOCPARSER_TOKEN")
        self.app_host = os.getenv("APP_HOST")
        self.pipeline_mode = cast(
            Literal["lightweight", "accurate"],
            _read_mode("PIPELINE_MODE", PIPELINE_MODES, "accurate"),
        )
        self.ocr_mode = cast(
            Literal["docling", "docparser"], _read_mode("OCR_MODE", OCR_MODES, "docling")
        )
        self.agentic_architecture = cast(
            Literal["custom", "langchain"],
            _read_mode("AGENTIC_ARCHITECTURE", AGENTIC_ARCHITECTURES, "custom"),
        )


config = Config()
