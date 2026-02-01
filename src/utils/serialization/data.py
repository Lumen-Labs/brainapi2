"""
File: /data.py
Created Date: Friday October 24th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday October 24th 2025 6:52:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import re


def str_to_json(text: str | None, empty_fallback: bool = False) -> list[str]:
    """
    Convert a string to a json list of strings.
    """
    if not text:
        return []
    text = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except Exception:
        try:
            arr_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if arr_match:
                json_text = arr_match.group(0)
            else:
                obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if obj_match:
                    json_text = obj_match.group(0)
                else:
                    json_text = text
            return json.loads(json_text)
        except Exception:
            if empty_fallback:
                return []
            raise Exception(f"Invalid JSON string: {text}")
