"""
File: /main.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 12:11:27 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.data import DataAdapter
from src.lib.mongo.client import _mongo_client

data_adapter = DataAdapter()
data_adapter.add_client(_mongo_client)
