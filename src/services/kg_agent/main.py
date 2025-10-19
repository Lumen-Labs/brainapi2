"""
File: /main.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:28:43 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.llm import LLMAdapter
from src.lib.llm.client_large import _llm_large_client


# Initialze the adapters
llm_large_adapter = LLMAdapter()
llm_large_adapter.add_client(_llm_large_client)
