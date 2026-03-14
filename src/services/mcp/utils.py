"""
File: /utils.py
Project: mcp
Created Date: Saturday March 14th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday March 14th 2026 3:56:40 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from src.config import config
from src.services.data.main import data_adapter


def guard_brainpat(pat: str = None, target_brain: str = None) -> bool | str:
    """
    Ensure the brainpat guarantees the access to the requested resource.

    If the BrainPAT is not matching any brain, it will try to match with the systempat,
    if the systempat is not matching, it will raise an auth error.
    """
    if pat == config.brainpat_token:
        return True

    brains = data_adapter.get_brains_list()
    for brain in brains:
        if brain.pat == pat:
            if target_brain and brain.name_key != target_brain:
                return False
            return brain.name_key

    return False
