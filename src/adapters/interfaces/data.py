"""
File: /data.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel

from src.constants.data import Brain, KGChanges, Observation, StructuredData, TextChunk


class SearchResult(BaseModel):
    """
    Search result model.
    """

    text_chunks: List[TextChunk]
    observations: List[Observation]


class DataClient(ABC):
    """
    Abstract base class for data clients.
    """

    @abstractmethod
    def save_text_chunk(self, text_chunk: TextChunk, brain_id: str) -> TextChunk:
        """
        Save data to the data client.
        """
        raise NotImplementedError("save method not implemented")

    @abstractmethod
    def save_observations(
        self, observations: List[Observation], brain_id: str
    ) -> Observation:
        """
        Save a list of observations to the data client.
        """
        raise NotImplementedError("save_observations method not implemented")

    @abstractmethod
    def search(self, text: str, brain_id: str) -> SearchResult:
        """
        Search data by text and return a list of text chunks and observations.
        """
        raise NotImplementedError("search method not implemented")

    @abstractmethod
    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool, brain_id: str
    ) -> Tuple[List[TextChunk], List[Observation]]:
        """
        Get data by their IDs.
        """
        raise NotImplementedError("get_by_ids method not implemented")

    @abstractmethod
    def save_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        """
        Save a structured data to the data client.
        """
        raise NotImplementedError("save_structured_data method not implemented")

    @abstractmethod
    def create_brain(self, name_key: str) -> Brain:
        """
        Create a new brain in the data client.
        """
        raise NotImplementedError("create_brain method not implemented")

    @abstractmethod
    def get_brain(self, name_key: str) -> Brain:
        """
        Get a brain from the data client.
        """
        raise NotImplementedError("get_brain method not implemented")

    @abstractmethod
    def get_brains_list(self) -> List[Brain]:
        """
        Get the list of brains from the data client.
        """
        raise NotImplementedError("get_brains_list method not implemented")

    @abstractmethod
    def save_kg_changes(self, kg_changes: KGChanges, brain_id: str) -> KGChanges:
        """
        Save a KG changes to the data client.
        """
        raise NotImplementedError("save_kg_changes method not implemented")

    @abstractmethod
    def get_structured_data_by_id(self, id: str, brain_id: str) -> StructuredData:
        """
        Get structured data by ID.
        """
        raise NotImplementedError("get_structured_data_by_id method not implemented")

    @abstractmethod
    def get_structured_data_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[StructuredData]:
        """
        Retrieve structured data entries for a specific brain with optional filtering and pagination.
        
        Parameters:
            brain_id (str): Identifier of the brain to query.
            limit (int): Maximum number of items to return.
            skip (int): Number of items to skip (offset) for pagination.
            types (list[str] | None): If provided, restrict results to these structured data types.
            query_text (str | None): If provided, filter entries that match the text query.
        
        Returns:
            list[StructuredData]: List of matching StructuredData objects.
        """
        raise NotImplementedError("get_structured_data_list method not implemented")

    @abstractmethod
    def get_structured_data_types(self, brain_id: str) -> list[str]:
        """
        Get all unique types from structured data.
        """
        raise NotImplementedError("get_structured_data_types method not implemented")

    @abstractmethod
    def get_observation_by_id(self, id: str, brain_id: str) -> Observation:
        """
        Get observation by ID.
        """
        raise NotImplementedError("get_observation_by_id method not implemented")

    @abstractmethod
    def get_observations_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        resource_id: str = None,
        labels: list[str] = None,
        query_text: str = None,
    ) -> list[Observation]:
        """
        Retrieve a filtered, paginated list of observations for a specific brain.
        
        Parameters:
            brain_id (str): Identifier of the brain to query.
            limit (int): Maximum number of observations to return.
            skip (int): Number of observations to skip (offset) for pagination.
            resource_id (str | None): If provided, restrict results to observations for this resource.
            labels (list[str] | None): If provided, include only observations that have at least one of these labels.
            query_text (str | None): If provided, include only observations whose text matches this query.
        
        Returns:
            observations (list[Observation]): A list of matching Observation objects ordered according to the backend's default sort.
        """
        raise NotImplementedError("get_observations_list method not implemented")

    @abstractmethod
    def get_observation_labels(self, brain_id: str) -> list[str]:
        """
        Retrieve all unique observation labels for the specified brain.
        
        Parameters:
            brain_id (str): Identifier of the brain whose observation labels should be retrieved.
        
        Returns:
            list[str]: Unique label strings present in the brain's observations.
        """
        raise NotImplementedError("get_observation_labels method not implemented")

    @abstractmethod
    def get_changelog_by_id(self, id: str, brain_id: str) -> KGChanges:
        """
        Retrieve a changelog entry by its identifier within a brain.

        Parameters:
            id (str): Identifier of the changelog to retrieve.
            brain_id (str): Identifier of the brain that contains the changelog.

        Returns:
            KGChanges: The changelog entry matching the provided `id` in the specified `brain_id`.
        """
        raise NotImplementedError("get_changelog_by_id method not implemented")

    @abstractmethod
    def get_changelogs_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[KGChanges]:
        """
        Retrieve a paginated list of knowledge-graph changelogs for a brain.

        Parameters:
            brain_id (str): Identifier of the brain whose changelogs to retrieve.
            limit (int): Maximum number of changelogs to return.
            skip (int): Number of changelogs to skip (offset) for pagination.
            types (list[str] | None): Optional list of changelog types to filter results by.
            query_text (str | None): Optional text to filter changelogs by relevance or content.

        Returns:
            list[KGChanges]: A list of KGChanges matching the provided filters and pagination.
        """
        raise NotImplementedError("get_changelogs_list method not implemented")

    @abstractmethod
    def get_changelog_types(self, brain_id: str) -> list[str]:
        """
        Return the unique changelog types present for the specified brain.
        
        Parameters:
            brain_id (str): Identifier of the brain to query.
        
        Returns:
            list[str]: List of unique changelog type names for the brain (order not guaranteed).
        """
        raise NotImplementedError("get_changelog_types method not implemented")

    @abstractmethod
    def update_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        """
        Update an existing StructuredData record for a specific brain.
        
        Parameters:
            structured_data (StructuredData): StructuredData object containing the updated content and identifier of the record to update.
            brain_id (str): Identifier of the brain that owns the structured data.
        
        Returns:
            StructuredData: The stored StructuredData after the update.
        """
        raise NotImplementedError("update_structured_data method not implemented")