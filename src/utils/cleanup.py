"""
File: /cleanup.py
Project: utils
Created Date: Saturday January 31st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 31st 2026 9:44:03 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""


def strip_properties(objs: list[dict], pop_also: list[str] = []) -> list[dict]:
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
