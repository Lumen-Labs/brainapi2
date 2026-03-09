import json
from typing import Any, Callable, Optional

from pydantic import BaseModel, TypeAdapter, ValidationError

from src.utils.cleanup import _last_json_object, _repair_trailing_commas, strip_json

from .schema_utils import get_single_list_field_name, validate_list_response_fallback


def parse_structured_output(
    n_message: Any,
    get_effective_schema: Callable[[], Any],
) -> Optional[BaseModel]:
    parsed = (
        strip_json(n_message) if isinstance(n_message, str) else n_message
    )
    if (
        isinstance(n_message, str)
        and not isinstance(parsed, list)
        and parsed == {}
    ):
        try:
            raw_parsed = json.loads(
                n_message.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            if isinstance(raw_parsed, list):
                parsed = raw_parsed
            elif isinstance(raw_parsed, dict):
                parsed = raw_parsed
        except Exception:
            pass
        if parsed == {} and "[" in n_message:
            start = n_message.index("[")
            depth = 0
            end = start
            for i in range(start, len(n_message)):
                if n_message[i] == "[":
                    depth += 1
                elif n_message[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                substring = n_message[start:end]
                for repair in (_repair_trailing_commas, lambda s: s):
                    try:
                        raw_parsed = json.loads(repair(substring))
                        if isinstance(raw_parsed, list):
                            parsed = raw_parsed
                            break
                    except Exception:
                        continue
        if parsed == {} and "{" in n_message:
            start = n_message.index("{")
            try:
                raw_parsed = json.loads(n_message[start:])
                if isinstance(raw_parsed, dict) and raw_parsed:
                    parsed = raw_parsed
            except Exception:
                pass
        if parsed == {} and isinstance(n_message, str):
            raw = (
                n_message.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            last_obj = _last_json_object(raw)
            if last_obj:
                parsed = last_obj
    effective = get_effective_schema()
    if isinstance(parsed, list):
        list_field = (
            get_single_list_field_name(effective) if effective else None
        )
        if list_field is not None:
            parsed = {list_field: parsed}
    if isinstance(parsed, dict) and len(parsed) == 1:
        list_field = (
            get_single_list_field_name(effective) if effective else None
        )
        if list_field is not None:
            single_key = next(iter(parsed))
            single_val = parsed[single_key]
            if isinstance(single_val, list) and single_key != list_field:
                parsed = {list_field: single_val}
    if not parsed:
        return None
    if effective is None:
        return None
    try:
        if hasattr(effective, "model_validate"):
            return effective.model_validate(parsed)
        return TypeAdapter(effective).validate_python(parsed)
    except ValidationError:
        list_field = get_single_list_field_name(effective)
        if (
            list_field
            and isinstance(parsed, dict)
            and len(parsed) == 1
            and isinstance(parsed.get(list_field), list)
        ):
            fallback = validate_list_response_fallback(
                effective, list_field, parsed[list_field]
            )
            if fallback is not None:
                return fallback
        raise
    return None


def finalize_structured_response(
    structured_response: Any,
    output_schema: Any,
    get_effective_schema: Callable[[], Any],
) -> Optional[BaseModel]:
    if not isinstance(structured_response, dict) or output_schema is None:
        return structured_response
    effective = get_effective_schema()
    if effective is None:
        return structured_response
    try:
        if hasattr(effective, "model_validate"):
            return effective.model_validate(structured_response)
        return TypeAdapter(effective).validate_python(structured_response)
    except ValidationError:
        list_field = get_single_list_field_name(effective)
        if (
            list_field
            and len(structured_response) == 1
            and isinstance(structured_response.get(list_field), list)
        ):
            fallback = validate_list_response_fallback(
                effective, list_field, structured_response[list_field]
            )
            if fallback is not None:
                return fallback
    except Exception:
        pass
    return structured_response
