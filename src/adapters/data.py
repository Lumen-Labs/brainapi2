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
from src.constants.data import Brain, Observation, StructuredData, TextChunk


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
        return self.data.save_text_chunk(text_chunk, brain_id)

    def save_observations(
        self, observations: List[Observation], brain_id: str = "default"
    ) -> Observation:
        """
        Save a list of observations to the data client.
        """
        return self.data.save_observations(observations, brain_id)

    def search(self, text: str, brain_id: str = "default") -> SearchResult:
        """
        Search data by text and return a list of text chunks and observations.
        """
        return self.data.search(text, brain_id)

    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool, brain_id: str = "default"
    ) -> Tuple[List[TextChunk], List[Observation]]:
        """
        Get data by their IDs.
        """
        return self.data.get_text_chunks_by_ids(ids, with_observations, brain_id)

    def save_structured_data(
        self, structured_data: StructuredData, brain_id: str = "default"
    ) -> StructuredData:
        """
        Save a structured data to the data client.
        """
        return self.data.save_structured_data(structured_data, brain_id)

    def create_brain(self, name_key: str) -> Brain:
        """
        Create a new brain in the data client.
        """
        return self.data.create_brain(name_key)

    def get_brain(self, name_key: str) -> Brain:
        """
        Get a brain from the data client.
        """
        return self.data.get_brain(name_key)
