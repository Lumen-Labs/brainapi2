"""
File: /logging.py
Created Date: Saturday November 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday November 1st 2025 5:28:02 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import logging
import os
import time
from typing import Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def log(*args, **kwargs):
    if os.getenv("DEBUG") == "true":
        message = " ".join(str(arg) for arg in args)
        if kwargs:
            message += " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(message)


def logt(fn: Callable, context: str = "", *args, **kwargs):
    st = time.time()
    result = fn(*args, **kwargs)
    et = time.time()
    if os.getenv("DEBUG") == "true":
        logger.info(f"[{fn.__name__} -> {et-st}s] {context}")
    return result
