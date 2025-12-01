"""
File: /data.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:01:13 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel

from src.constants.data import Brain, Observation, StructuredData, TextChunk


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
