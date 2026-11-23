"""Opt-in spreadsheet experiment brain gate."""

import os

FLAG = "SPREADSHEET_EXPERIMENT_BRAINS"


def spreadsheet_experiment_brains(raw=None):
    text = os.getenv(FLAG, "") if raw is None else raw
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def spreadsheet_experiment_enabled(brain_id, raw=None):
    if not brain_id:
        return False
    brains = spreadsheet_experiment_brains(raw)
    return "*" in brains or brain_id in brains
