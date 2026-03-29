"""
File: /main.py
Created Date: Thursday October 23rd 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday October 23rd 2025 10:19:44 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from src.core.agents.observations_agent import ObservationsAgent
from src.core.instances import llm_small_adapter


observations_agent = ObservationsAgent(llm_small_adapter)
