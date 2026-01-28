"""
File: /client.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
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
        self, ids: List[str], with_observations: bool = False, brain_id: str = "default"
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
        collection = self.get_collection("brains", config.mongo.system_database)
        brain = Brain(name_key=name_key)
        brain_dict = brain.model_dump(mode="json", exclude={"id"})
        result = collection.insert_one(brain_dict)
        brain.id = str(result.inserted_id)
        return brain

    def get_brain(self, name_key: str) -> Brain:
        collection = self.get_collection("brains", config.mongo.system_database)
        result = collection.find_one({"name_key": name_key})
        if not result:
            return None
        return Brain(
            id=str(result["_id"]),
            name_key=result["name_key"],
            pat=result.get("pat"),
        )

    def get_brains_list(self) -> List[Brain]:
        collection = self.get_collection("brains", config.mongo.system_database)
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

    def get_structured_data_by_id(self, id: str, brain_id: str) -> StructuredData:
        collection = self.get_collection("structured_data", database=brain_id)
        result = collection.find_one({"id": id})
        if not result:
            return None
        return StructuredData(
            id=result["id"],
            data=result["data"],
            types=result["types"],
            metadata=result.get("metadata", None),
            inserted_at=result.get("inserted_at", None),
        )

    def get_structured_data_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[StructuredData]:
        collection = self.get_collection("structured_data", database=brain_id)
        query = {}
        if types:
            query["types"] = {"$in": types}

        if query_text:
            query["$or"] = [
                {"data": {"$regex": query_text, "$options": "i"}},
                {"types": {"$regex": query_text, "$options": "i"}},
                {"metadata": {"$regex": query_text, "$options": "i"}},
            ]

        results = collection.find(query).skip(skip).limit(limit)
        return [
            StructuredData(
                id=result["id"],
                data=result["data"],
                types=result["types"],
                metadata=result.get("metadata", None),
                inserted_at=result.get("inserted_at", None),
            )
            for result in results
        ]

    def get_structured_data_types(self, brain_id: str) -> list[str]:
        collection = self.get_collection("structured_data", database=brain_id)
        pipeline = [
            {"$unwind": "$types"},
            {"$group": {"_id": "$types"}},
            {"$sort": {"_id": 1}},
        ]
        results = collection.aggregate(pipeline)
        return [result["_id"] for result in results]

    def get_observation_by_id(self, id: str, brain_id: str) -> Observation:
        collection = self.get_collection("observations", database=brain_id)
        result = collection.find_one({"id": id})
        if not result:
            return None
        return Observation(
            id=result["id"],
            text=result["text"],
            metadata=result.get("metadata", None),
            resource_id=result["resource_id"],
            inserted_at=result.get("inserted_at", None),
        )

    def get_observations_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        resource_id: str = None,
        labels: list[str] = None,
        query_text: str = None,
    ) -> list[Observation]:
        collection = self.get_collection("observations", database=brain_id)
        query = {}
        if resource_id:
            query["resource_id"] = resource_id
        if labels:
            query["metadata.labels"] = {"$in": labels}

        if query_text:
            query["$or"] = [
                {"text": {"$regex": query_text, "$options": "i"}},
                {"resource_id": {"$regex": query_text, "$options": "i"}},
                {"metadata": {"$regex": query_text, "$options": "i"}},
            ]

        results = collection.find(query).skip(skip).limit(limit)
        return [
            Observation(
                id=result["id"],
                text=result["text"],
                metadata=result.get("metadata", None),
                resource_id=result["resource_id"],
                inserted_at=result.get("inserted_at", None),
            )
            for result in results
        ]

    def get_observation_labels(self, brain_id: str) -> list[str]:
        """
        Return the list of unique observation labels for the specified brain, sorted lexicographically.

        Parameters:
            brain_id (str): Name/key of the brain used as the MongoDB database to query.

        Returns:
            list[str]: Sorted list of distinct label strings found in observations' metadata.labels.
        """
        collection = self.get_collection("observations", database=brain_id)
        pipeline = [
            {"$match": {"metadata.labels": {"$exists": True}}},
            {"$unwind": "$metadata.labels"},
            {"$group": {"_id": "$metadata.labels"}},
            {"$sort": {"_id": 1}},
        ]
        results = collection.aggregate(pipeline)
        return [result["_id"] for result in results]

    def get_changelog_by_id(self, id: str, brain_id: str) -> KGChanges:
        """
        Retrieve a changelog entry by id from the specified brain's kg_changes collection.

        If a document is found, convert string values of the nested `change.type` and the top-level `type`
        fields to the `KGChangesType` enum when present, then validate and return a `KGChanges` model.
        Returns `None` if no matching document exists.

        Parameters:
            id (str): The changelog document id to retrieve.
            brain_id (str): The brain (database) name where the kg_changes collection resides.

        Returns:
            KGChanges or None: The validated `KGChanges` model for the found document, or `None` if not found.
        """
        from src.constants.data import KGChangesType

        collection = self.get_collection("kg_changes", database=brain_id)
        result = collection.find_one({"id": id})
        if not result:
            return None

        if "change" in result and isinstance(result["change"], dict):
            if "type" in result["change"] and isinstance(result["change"]["type"], str):
                result["change"]["type"] = KGChangesType(result["change"]["type"])

        if "type" in result and isinstance(result["type"], str):
            result["type"] = KGChangesType(result["type"])

        return KGChanges.model_validate(result)

    def get_changelogs_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[KGChanges]:
        """
        Retrieve changelog entries for a brain with optional type filtering and text search.

        Parameters:
            brain_id (str): Identifier of the brain (database) to query.
            limit (int): Maximum number of changelogs to return.
            skip (int): Number of changelogs to skip (offset).
            types (list[str] | None): If provided, restrict results to changelogs whose top-level `type` is in this list.
            query_text (str | None): If provided, perform a case-insensitive substring search on the `change` and `type` fields.

        Returns:
            list[KGChanges]: Changelogs validated into `KGChanges` models, sorted by `timestamp` descending. Entries that fail model validation are skipped (an error message is printed). String values in nested `change.type` and top-level `type` are converted to `KGChangesType` when applicable.
        """
        from src.constants.data import KGChangesType

        collection = self.get_collection("kg_changes", database=brain_id)
        query = {}
        if types:
            query["type"] = {"$in": types}

        if query_text:
            query["$or"] = [
                {"change.subject.name": {"$regex": query_text, "$options": "i"}},
                {"change.object.name": {"$regex": query_text, "$options": "i"}},
                {"change.predicate.name": {"$regex": query_text, "$options": "i"}},
                {"change.node.name": {"$regex": query_text, "$options": "i"}},
                {"type": {"$regex": query_text, "$options": "i"}},
            ]

        results = collection.find(query).skip(skip).limit(limit).sort("timestamp", -1)
        changelogs = []
        for result in results:
            try:
                if "change" in result and isinstance(result["change"], dict):
                    if "type" in result["change"] and isinstance(
                        result["change"]["type"], str
                    ):
                        result["change"]["type"] = KGChangesType(
                            result["change"]["type"]
                        )

                if "type" in result and isinstance(result["type"], str):
                    result["type"] = KGChangesType(result["type"])

                changelog = KGChanges.model_validate(result)
                changelogs.append(changelog)
            except Exception as e:
                print(f"Error parsing changelog {result.get('id')}: {e}")
                continue

        return changelogs

    def get_changelog_types(self, brain_id: str) -> list[str]:
        """
        Retrieve the sorted list of distinct changelog types for a brain.

        Parameters:
            brain_id (str): Identifier of the brain (database name) to query.

        Returns:
            list[str]: Changelog type strings sorted in ascending order.
        """
        collection = self.get_collection("kg_changes", database=brain_id)
        pipeline = [{"$group": {"_id": "$type"}}, {"$sort": {"_id": 1}}]
        results = collection.aggregate(pipeline)
        return [result["_id"] for result in results]

    def update_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        """
        Update an existing StructuredData record in the specified brain's structured_data collection.
        
        Parameters:
            structured_data (StructuredData): The structured data object whose stored record will be updated; its `id` is used to locate the document.
            brain_id (str): Identifier of the brain (database) containing the structured_data collection to update.
        
        Returns:
            StructuredData: The same `structured_data` object that was passed in.
        
        Notes:
            The function updates the document matching `structured_data.id` by setting its fields to those from `structured_data`. If no matching document exists, no document is inserted.
        """
        collection = self.get_collection("structured_data", database=brain_id)
        collection.update_one(
            {"id": structured_data.id},
            {"$set": structured_data.model_dump(mode="json")},
        )
        return structured_data


_mongo_client = MongoClient()