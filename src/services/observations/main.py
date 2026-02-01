"""
File: /main.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 10:19:44 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from src.adapters.llm import LLMAdapter
from src.core.agents.observations_agent import ObservationsAgent
from src.lib.llm.client_small import _llm_small_client


llm_adapter = LLMAdapter()
llm_adapter.add_client(_llm_small_client)

observations_agent = ObservationsAgent(llm_adapter)
