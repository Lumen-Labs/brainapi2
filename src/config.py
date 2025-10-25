"""
File: /config.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:45:58 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

# pylint: disable=too-few-public-methods

import os
import dotenv

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


class RedisConfig:
    """
    Configuration class for the Redis configuration.
    """

    def __init__(self):
        self.host = os.getenv("REDIS_HOST")
        self.port = os.getenv("REDIS_PORT")

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
        self.port = os.getenv("MILVUS_PORT")

        if [self.host, self.port].count(None) > 0:
            raise ValueError("Milvus configuration is not complete")


class EmbeddingsConfig:
    """
    Configuration class for the Embeddings configuration.
    """

    def __init__(self):
        self.full_endpoint = os.getenv("EMBEDDINGS_FULL_ENDPOINT")
        self.key = os.getenv("EMBEDDINGS_KEY")


class MongoConfig:
    """
    Configuration class for the Mongo configuration.
    """

    def __init__(self):
        self.host = os.getenv("MONGO_HOST")
        self.port = int(os.getenv("MONGO_PORT"))
        self.username = os.getenv("MONGO_USERNAME")
        self.password = os.getenv("MONGO_PASSWORD")

        if [self.host, self.port, self.username, self.password].count(None) > 0:
            raise ValueError("Mongo configuration is not complete")


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


config = Config()
