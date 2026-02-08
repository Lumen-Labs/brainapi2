"""
File: /data.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List, Tuple
from src.adapters.interfaces.data import DataClient, SearchResult
from src.constants.data import Brain, KGChanges, Observation, StructuredData, TextChunk


class DataAdapter:
    """
    Data adapter.
    """

    def __init__(self):
        self.data = None

    def add_client(self, client: DataClient) -> None:
        """
        Add a data client to the adapter.
        """
        self.data = client

    def save_text_chunk(
        self, text_chunk: TextChunk, brain_id: str = "default"
    ) -> TextChunk:
        """
        Save a text chunk to the data client.
        """
        return self.data.save_text_chunk(text_chunk=text_chunk, brain_id=brain_id)

    def save_observations(
        self, observations: List[Observation], brain_id: str = "default"
    ) -> Observation:
        """
        Save a list of observations to the data client.
        """
        return self.data.save_observations(observations=observations, brain_id=brain_id)

    def search(self, text: str, brain_id: str = "default") -> SearchResult:
        """
        Search data by text and return a list of text chunks and observations.
        """
        return self.data.search(text=text, brain_id=brain_id)

    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool, brain_id: str = "default"
    ) -> Tuple[List[TextChunk], List[Observation]]:
        """
        Get data by their IDs.
        """
        return self.data.get_text_chunks_by_ids(
            ids=ids, with_observations=with_observations, brain_id=brain_id
        )

    def save_structured_data(
        self, structured_data: StructuredData, brain_id: str = "default"
    ) -> StructuredData:
        """
        Save a structured data to the data client.
        """
        return self.data.save_structured_data(
            structured_data=structured_data, brain_id=brain_id
        )

    def create_brain(self, name_key: str) -> Brain:
        """
        Create a new brain in the data client.
        """
        return self.data.create_brain(name_key=name_key)

    def get_brain(self, name_key: str) -> Brain:
        """
        Get a brain from the data client.
        """
        return self.data.get_brain(name_key=name_key)

    def get_brains_list(self) -> List[Brain]:
        """
        Get the list of brains from the data client.
        """
        return self.data.get_brains_list()

    def save_kg_changes(
        self, kg_changes: KGChanges, brain_id: str = "default"
    ) -> KGChanges:
        """
        Save a KG changes to the data client.
        """
        return self.data.save_kg_changes(kg_changes=kg_changes, brain_id=brain_id)

    def get_structured_data_by_id(
        self, id: str, brain_id: str = "default"
    ) -> StructuredData:
        """
        Get structured data by ID.
        """
        return self.data.get_structured_data_by_id(id=id, brain_id=brain_id)

    def get_structured_data_list(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[StructuredData]:
        """
        Get a list of structured data.
        """
        return self.data.get_structured_data_list(
            brain_id=brain_id,
            limit=limit,
            skip=skip,
            types=types,
            query_text=query_text,
        )

    def get_structured_data_types(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique types from structured data.
        """
        return self.data.get_structured_data_types(brain_id=brain_id)

    def get_observation_by_id(self, id: str, brain_id: str = "default") -> Observation:
        """
        Get observation by ID.
        """
        return self.data.get_observation_by_id(id=id, brain_id=brain_id)

    def get_observations_list(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        resource_id: str = None,
        labels: list[str] = None,
        query_text: str = None,
    ) -> list[Observation]:
        """
        Get a list of observations.
        """
        return self.data.get_observations_list(
            brain_id=brain_id,
            limit=limit,
            skip=skip,
            resource_id=resource_id,
            labels=labels,
            query_text=query_text,
        )

    def get_observation_labels(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique observation labels for the specified brain.

        Parameters:
            brain_id (str): Identifier of the brain to query.

        Returns:
            list[str]: A list of unique observation label strings for the specified brain.
        """
        return self.data.get_observation_labels(brain_id=brain_id)

    def get_changelog_by_id(self, id: str, brain_id: str = "default") -> KGChanges:
        """
        Retrieve a changelog entry by its identifier.

        Parameters:
            id (str): Identifier of the changelog entry to retrieve.
            brain_id (str): Brain namespace key to query; defaults to "default".

        Returns:
            KGChanges: The changelog entry matching the given `id`.
        """
        return self.data.get_changelog_by_id(id=id, brain_id=brain_id)

    def get_changelogs_list(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[KGChanges]:
        """
        Retrieve a paginated list of knowledge-graph changelogs for a brain.

        Parameters:
            brain_id (str): Identifier of the brain to query.
            limit (int): Maximum number of changelogs to return.
            skip (int): Number of changelogs to skip (offset).
            types (list[str] | None): If provided, restrict results to these changelog types.
            query_text (str | None): If provided, filter changelogs by matching text.

        Returns:
            list[KGChanges]: Changelogs matching the filters and pagination parameters.
        """
        return self.data.get_changelogs_list(
            brain_id=brain_id,
            limit=limit,
            skip=skip,
            types=types,
            query_text=query_text,
        )

    def get_changelog_types(self, brain_id: str = "default") -> list[str]:
        """
        Retrieve distinct changelog types for a brain.

        Parameters:
            brain_id (str): Identifier of the brain to query; defaults to "default".

        Returns:
            list[str]: List of changelog type names.
        """
        return self.data.get_changelog_types(brain_id=brain_id)

    def update_structured_data(
        self, structured_data: StructuredData, brain_id: str = "default"
    ) -> StructuredData:
        """
        Update an existing structured data entry.

        Parameters:
                structured_data (StructuredData): The structured data object with updated information.
                brain_id (str): Identifier of the brain context for the update.

        Returns:
                StructuredData: The updated structured data object.
        """
        return self.data.update_structured_data(
            structured_data=structured_data, brain_id=brain_id
        )

    def get_last_text_chunks(
        self, brain_id: str = "default", limit: int = 10
    ) -> list[TextChunk]:
        """
        Get the last text chunks from the data client.
        """
        return self.data.get_last_text_chunks(brain_id=brain_id, limit=limit)

    def get_last_structured_data(
        self, brain_id: str = "default", limit: int = 10
    ) -> list[StructuredData]:
        """
        Get the last structured data from the data client.
        """
        return self.data.get_last_structured_data(brain_id=brain_id, limit=limit)
