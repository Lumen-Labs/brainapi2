from typing import Any, Optional, Type, get_args, get_origin

from pydantic import BaseModel, TypeAdapter


def get_effective_output_schema(output_schema: Any):
    if output_schema is None:
        return None
    effective = getattr(output_schema, "schema", output_schema)
    if callable(effective):
        effective = output_schema
    return effective


def get_output_schema_json_schema(output_schema: Any) -> Optional[dict]:
    effective = get_effective_output_schema(output_schema)
    if effective is None:
        return None
    try:
        if hasattr(effective, "model_json_schema"):
            return effective.model_json_schema()
        return TypeAdapter(effective).json_schema()
    except (Exception, NameError):
        return None


def validate_list_response_fallback(
    effective: Type[BaseModel], list_field: str, items: list
) -> Optional[BaseModel]:
    if not getattr(effective, "model_fields", None):
        return None
    field_info = effective.model_fields.get(list_field)
    if field_info is None:
        return None
    ann = getattr(field_info, "annotation", None)
    if get_origin(ann) is not list:
        return None
    args = get_args(ann)
    if not args:
        return None
    item_type = args[0]
    if not isinstance(item_type, type):
        return None
    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            if hasattr(item_type, "model_validate"):
                validated.append(item_type.model_validate(item))
            else:
                validated.append(TypeAdapter(item_type).validate_python(item))
        except Exception:
            try:
                allowed = set(getattr(item_type, "model_fields", {}).keys())
                filtered = {k: v for k, v in item.items() if k in allowed}
                if allowed and filtered:
                    if hasattr(item_type, "model_validate"):
                        validated.append(item_type.model_validate(filtered))
                    else:
                        validated.append(
                            TypeAdapter(item_type).validate_python(filtered)
                        )
            except Exception:
                pass
    if not validated:
        return None
    try:
        return effective.model_validate({list_field: validated})
    except Exception:
        try:
            return effective.model_construct(**{list_field: validated})
        except Exception:
            return None


def flatten_json_schema_for_llm(schema: dict) -> dict:
    def resolve_refs(obj: dict, defs: dict) -> dict:
        out = {}
        for k, v in obj.items():
            if k == "description":
                continue
            if k == "$ref":
                ref = v.split("/")[-1]
                return resolve_refs(defs.get(ref, {}).copy(), defs)
            if isinstance(v, dict):
                out[k] = resolve_refs(v, defs)
            elif isinstance(v, list):
                out[k] = [
                    resolve_refs(x, defs) if isinstance(x, dict) else x for x in v
                ]
            else:
                out[k] = v
        return out

    def drop_internal_keys(obj: dict) -> dict:
        out = {}
        for k, v in obj.items():
            if k in ("$defs", "$ref", "strict"):
                continue
            if k == "title" and isinstance(v, str) and v.startswith("_"):
                continue
            if isinstance(v, dict):
                out[k] = drop_internal_keys(v)
            elif isinstance(v, list):
                out[k] = [
                    drop_internal_keys(x) if isinstance(x, dict) else x for x in v
                ]
            else:
                out[k] = v
        return out

    copy = dict(schema)
    defs = copy.pop("$defs", {})
    resolved = resolve_refs(copy, defs)
    return drop_internal_keys(resolved)


def get_single_list_field_name(schema: Any) -> Optional[str]:
    if schema is None:
        return None
    if hasattr(schema, "model_fields"):
        for name, field in schema.model_fields.items():
            ann = getattr(field, "annotation", None)
            if get_origin(ann) is list:
                return name
    try:
        if hasattr(schema, "model_json_schema"):
            js = schema.model_json_schema()
        else:
            js = TypeAdapter(schema).json_schema()
    except (Exception, NameError):
        return None
    defs = js.get("$defs") or {}
    props = js.get("properties") or {}
    if len(props) != 1:
        return None
    for name, prop in props.items():
        if isinstance(prop, dict) and prop.get("type") == "array":
            return name
        if isinstance(prop, dict) and "$ref" in prop:
            ref = prop["$ref"].split("/")[-1]
            ref_schema = defs.get(ref, {})
            if isinstance(ref_schema, dict) and ref_schema.get("type") == "array":
                return name
    return None
