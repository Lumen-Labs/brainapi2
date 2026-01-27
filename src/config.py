"""
File: /config.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

# pylint: disable=too-few-public-methods

import os
import dotenv
from pathlib import Path

dotenv.load_dotenv(
    dotenv_path=f".env{'.' + os.getenv('ENV') if os.getenv('ENV') else ''}"
)


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

    def __init__(self):
        self.full_endpoint = os.getenv("EMBEDDINGS_FULL_ENDPOINT")
        self.key = os.getenv("EMBEDDINGS_KEY")

        self.small_model = os.getenv("EMBEDDINGS_SMALL_MODEL")

        if [self.full_endpoint, self.key, self.small_model].count(None) > 0:
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
        self.worker_concurrency = os.getenv("CELERY_WORKER_CONCURRENCY")
        if [self.worker_concurrency].count(None) > 0:
            raise ValueError("Celery configuration is not complete")


class PricingConfig:
    """
    Configuration class for the Pricing configuration.
    """

    def __init__(self):
        self.input_token_price = float(os.getenv("INPUT_TOKEN_PRICE", 0))
        self.output_token_price = float(os.getenv("OUTPUT_TOKEN_PRICE", 0))


class Config:
    """
    Configuration class for the application.
    """

    def __init__(self):
        self.azure = AzureConfig()
        self.redis = RedisConfig()
        self.neo4j = Neo4jConfig()
        self.milvus = MilvusConfig()
        self.embeddings = EmbeddingsConfig()
        self.mongo = MongoConfig()
        self.gcp = GCPConfig()
        self.celery = CeleryConfig()
        self.pricing = PricingConfig()

        self.run_graph_consolidator = (
            os.getenv("RUN_GRAPH_CONSOLIDATOR", "true") == "true"
        )
        self.brainpat_token = os.getenv("BRAINPAT_TOKEN")

        if not self.brainpat_token:
            raise ValueError("BrainPAT token is not set")


config = Config()
