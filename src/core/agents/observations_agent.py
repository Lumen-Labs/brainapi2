"""
File: /observations_agent.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 10:11:00 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
from src.adapters.llm import LLMAdapter
from src.constants.prompts.observations_agent import OBSERVATIONS_AGENT_SYSTEM_PROMPT
from src.utils.serialization.data import str_to_json


class ObservationsAgent:
    """
    Observations Agent. Used to observe the world and update the knowledge graph with new information.
    """

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter

    def observe(self, text: str, observate_for: Optional[List[str]]) -> str:
        """
        Observe the world and update the knowledge graph with new information.
        """
        return str_to_json(
            self.llm_adapter.generate_text(
                OBSERVATIONS_AGENT_SYSTEM_PROMPT.format(
                    text=text, observate_for=observate_for
                )
            ),
            empty_fallback=True,
        )
