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

from src.constants.data import Observation, StructuredData, TextChunk


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
    def save_text_chunk(self, text_chunk: TextChunk) -> TextChunk:
        """
        Save data to the data client.
        """
        raise NotImplementedError("save method not implemented")

    @abstractmethod
    def save_observations(self, observations: List[Observation]) -> Observation:
        """
        Save a list of observations to the data client.
        """
        raise NotImplementedError("save_observations method not implemented")

    @abstractmethod
    def search(self, text: str) -> SearchResult:
        """
        Search data by text and return a list of text chunks and observations.
        """
        raise NotImplementedError("search method not implemented")

    @abstractmethod
    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool
    ) -> Tuple[List[TextChunk], List[Observation]]:
        """
        Get data by their IDs.
        """
        raise NotImplementedError("get_by_ids method not implemented")

    @abstractmethod
    def save_structured_data(self, structured_data: StructuredData) -> StructuredData:
        """
        Save a structured data to the data client.
        """
        raise NotImplementedError("save_structured_data method not implemented")
