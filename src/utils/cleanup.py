"""
File: /cleanup.py
Project: utils
Created Date: Saturday January 31st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import re


def _repair_array_json(s: str) -> str:
    s = s.replace(',"{', ",{")
    s = re.sub(r'\}{2,}\s*,\s*"', '},"', s)
    return s


def _repair_trailing_commas(s: str) -> str:
    s = re.sub(r',\s*]', ']', s)
    s = re.sub(r',\s*}', '}', s)
    return s


def _extract_json_array_after_key(text: str, key: str):
    pattern = re.escape(key) + r"\s*:\s*\["
    match = re.search(pattern, text)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                substring = text[start : i + 1]
                try:
                    return json.loads(substring)
                except Exception:
                    try:
                        return json.loads(_repair_array_json(substring))
                    except Exception:
                        return None
        i += 1
    return None


def strip_properties(
    objs: list[dict], pop_also: list[str] | None = None
) -> list[dict]:
    if pop_also is None:
        pop_also = []
    cleaned_objs = []
    for obj in objs:
        if not isinstance(obj, dict):
            cleaned_objs.append(obj)
            continue
        cleaned_obj = obj.copy()

        keys_to_pop = []
        for key, value in cleaned_obj.items():
            if key in pop_also:
                keys_to_pop.append(key)
                continue

            # Python's bool type are subclasses of int, False is 0 and True is 1.
            # This can lead to unexpected behavior if '0' or '1' are used as keys.
            if isinstance(key, bool):
                continue

            if isinstance(value, dict):
                if len(value.keys()) == 0:
                    keys_to_pop.append(key)
                else:
                    cleaned_obj[key] = strip_properties([value], pop_also)[0]
            elif isinstance(value, list):
                if len(value) == 0:
                    keys_to_pop.append(key)
                else:
                    new_list = []
                    for item in value:
                        if isinstance(item, dict):
                            new_list.append(strip_properties([item], pop_also)[0])
                        else:
                            new_list.append(item)
                    cleaned_obj[key] = new_list
            elif isinstance(value, str):
                if len(value.strip()) == 0:
                    keys_to_pop.append(key)
            elif value is None:
                keys_to_pop.append(key)

        for key in keys_to_pop:
            cleaned_obj.pop(key)

        cleaned_objs.append(cleaned_obj)

    return cleaned_objs


def _last_json_object(text: str) -> dict:
    if not text or "{" not in text:
        return {}
    start = text.rfind("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    try:
                        return json.loads(_repair_array_json(text[start : i + 1]))
                    except Exception:
                        pass
                return {}
    return {}


def strip_json(text: str) -> dict:
    if not text:
        return {}
    if not isinstance(text, str):
        return {}
    raw = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"entities": parsed}
    except Exception:
        pass
    try:
        repaired = _repair_array_json(raw)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"entities": parsed}
    except Exception:
        pass
    try:
        repaired = _repair_trailing_commas(raw)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"entities": parsed}
    except Exception:
        pass
    if raw.strip().startswith("["):
        depth = 0
        start = raw.index("[")
        end = start
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            substring = raw[start:end]
            for repair in (lambda s: s, _repair_trailing_commas, _repair_array_json):
                try:
                    parsed = json.loads(repair(substring))
                    if isinstance(parsed, list):
                        return {"entities": parsed}
                except Exception:
                    continue
    obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            try:
                return json.loads(_repair_array_json(obj_match.group(0)))
            except Exception:
                pass
    entities = _extract_json_array_after_key(raw, "entities")
    if entities is not None:
        return {"entities": entities}
    last = _last_json_object(raw)
    if last and isinstance(last.get("entities"), list):
        return last
    return {}
