"""
File: /client.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 11:55:47 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List
from pymongo import MongoClient as PyMongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from src.adapters.interfaces.data import DataClient, SearchResult
from src.config import config
from src.constants.data import Brain, KGChanges, Observation, StructuredData, TextChunk


class MongoClient(DataClient):
    """
    Mongo client.
    """

    def __init__(self):
        if (
            hasattr(config.mongo, "connection_string")
            and config.mongo.connection_string
        ):
            self.client = PyMongoClient(config.mongo.connection_string)
        else:
            self.client = PyMongoClient(
                config.mongo.host,
                config.mongo.port,
                username=config.mongo.username,
                password=config.mongo.password,
                authSource="admin",
            )

    def get_database(self, database: str = "default") -> Database:
        return self.client[database]

    def get_collection(self, collection: str, database: str = "default") -> Collection:
        return self.get_database(database)[collection]

    def save_text_chunk(self, text_chunk: TextChunk, brain_id: str) -> TextChunk:
        collection = self.get_collection("text_chunks", database=brain_id)
        collection.insert_one(text_chunk.model_dump(mode="json"))
        return text_chunk

    def save_observations(
        self, observations: List[Observation], brain_id: str
    ) -> Observation:
        collection = self.get_collection("observations", database=brain_id)
        collection.insert_many([o.model_dump(mode="json") for o in observations])
        return observations

    def search(
        self, text: str, brain_id: str, collection: str = "*", limit: int = 10
    ) -> SearchResult:
        chunks_collection = self.get_collection(collection, database=brain_id)
        observations_collection = self.get_collection("observations", database=brain_id)

        text_chunks_results = chunks_collection.find(
            {"text": {"$regex": text, "$options": "i"}}
        )
        text_chunks = [
            TextChunk(
                id=result["_id"],
                text=result["text"],
                metadata=result.get("metadata", None),
            )
            for result in text_chunks_results
        ]
        observations = observations_collection.find(
            {"resource_id": {"$in": [text_chunk.id for text_chunk in text_chunks]}}
        )
        observations = [
            Observation(
                id=result["id"],
                text=result["text"],
                metadata=result.get("metadata", None),
                resource_id=result["resource_id"],
            )
            for result in observations
        ]

        return SearchResult(text_chunks=text_chunks, observations=observations)

    def get_text_chunks_by_ids(
        self, ids: List[str], brain_id: str, with_observations: bool = False
    ) -> List[TextChunk]:
        chunks_collection = self.get_collection("text_chunks", database=brain_id)
        text_chunks = chunks_collection.find({"id": {"$in": ids}})

        if with_observations:
            observations_collection = self.get_collection(
                "observations", database=brain_id
            )
            observations = observations_collection.find({"resource_id": {"$in": ids}})

        return (
            [
                TextChunk(
                    id=result["id"],
                    text=result["text"],
                    metadata=result.get("metadata", None),
                    inserted_at=result["inserted_at"],
                )
                for result in text_chunks
            ],
            (
                [
                    Observation(
                        id=result["id"],
                        text=result["text"],
                        metadata=result.get("metadata", None),
                        resource_id=result["resource_id"],
                        inserted_at=result["inserted_at"],
                    )
                    for result in observations
                ]
                if with_observations
                else []
            ),
        )

    def save_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        collection = self.get_collection("structured_data", database=brain_id)
        collection.insert_one(structured_data.model_dump(mode="json"))
        return structured_data

    def create_brain(self, name_key: str) -> Brain:
        collection = self.get_collection("brains", "system")
        brain = Brain(name_key=name_key)
        brain_dict = brain.model_dump(mode="json", exclude={"id"})
        result = collection.insert_one(brain_dict)
        brain.id = str(result.inserted_id)
        return brain

    def get_brain(self, name_key: str) -> Brain:
        collection = self.get_collection("brains", "system")
        result = collection.find_one({"name_key": name_key})
        if not result:
            return None
        return Brain(
            id=str(result["_id"]),
            name_key=result["name_key"],
            pat=result.get("pat"),
        )

    def get_brains_list(self) -> List[Brain]:
        collection = self.get_collection("brains", "system")
        result = collection.find()
        return [
            Brain(
                id=str(result["_id"]),
                name_key=result["name_key"],
                pat=result.get("pat"),
            )
            for result in result
        ]

    def save_kg_changes(self, kg_changes: KGChanges, brain_id: str) -> KGChanges:
        collection = self.get_collection("kg_changes", database=brain_id)
        collection.insert_one(kg_changes.model_dump(mode="json"))
        return kg_changes


_mongo_client = MongoClient()
