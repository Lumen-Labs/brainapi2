"""
File: /data.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:00:42 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Tuple
from src.adapters.interfaces.data import DataClient, SearchResult
from src.constants.data import Observation, StructuredData, TextChunk


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

    def save_text_chunk(self, text_chunk: TextChunk) -> TextChunk:
        """
        Save a text chunk to the data client.
        """
        return self.data.save_text_chunk(text_chunk)

    def save_observations(self, observations: List[Observation]) -> Observation:
        """
        Save a list of observations to the data client.
        """
        return self.data.save_observations(observations)

    def search(self, text: str) -> SearchResult:
        """
        Search data by text and return a list of text chunks and observations.
        """
        return self.data.search(text)

    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool
    ) -> Tuple[List[TextChunk], List[Observation]]:
        """
        Get data by their IDs.
        """
        return self.data.get_text_chunks_by_ids(ids, with_observations)

    def save_structured_data(self, structured_data: StructuredData) -> StructuredData:
        """
        Save a structured data to the data client.
        """
        return self.data.save_structured_data(structured_data)
