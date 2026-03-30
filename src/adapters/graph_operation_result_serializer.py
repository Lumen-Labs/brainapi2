import json
from abc import ABC, abstractmethod
from typing import Any, Iterable


class GraphOperationResultSerializer(ABC):
    @abstractmethod
    def can_handle(self, result: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, result: Any) -> str:
        raise NotImplementedError


class NoneResultSerializer(GraphOperationResultSerializer):
    def can_handle(self, result: Any) -> bool:
        return result is None

    def serialize(self, result: Any) -> str:
        return ""


class StringResultSerializer(GraphOperationResultSerializer):
    def can_handle(self, result: Any) -> bool:
        return isinstance(result, str)

    def serialize(self, result: Any) -> str:
        return result


class Neo4jResultSerializer(GraphOperationResultSerializer):
    def can_handle(self, result: Any) -> bool:
        return hasattr(result, "records")

    def serialize(self, result: Any) -> str:
        records = result.records or []
        limited_records = records[:20]
        serialized_records = []
        for record in limited_records:
            if hasattr(record, "data"):
                serialized_records.append(record.data())
                continue
            try:
                serialized_records.append(dict(record))
            except Exception:
                serialized_records.append(str(record))
        keys = None
        try:
            keys = list(result.keys())
        except Exception:
            try:
                keys = list(result.keys)
            except Exception:
                keys = None
        payload = {
            "records": serialized_records,
            "truncated": len(records) > 20,
        }
        if keys is not None:
            payload["keys"] = keys
        return json.dumps(payload, default=str)


class JsonResultSerializer(GraphOperationResultSerializer):
    def can_handle(self, result: Any) -> bool:
        return isinstance(result, (dict, list, tuple, int, float, bool))

    def serialize(self, result: Any) -> str:
        return json.dumps(result, default=str)


class FallbackResultSerializer(GraphOperationResultSerializer):
    def can_handle(self, result: Any) -> bool:
        return True

    def serialize(self, result: Any) -> str:
        return str(result)


class GraphOperationResultSerializerChain:
    def __init__(
        self, serializers: Iterable[GraphOperationResultSerializer] | None = None
    ):
        self.serializers = list(
            serializers
            or [
                NoneResultSerializer(),
                StringResultSerializer(),
                Neo4jResultSerializer(),
                JsonResultSerializer(),
                FallbackResultSerializer(),
            ]
        )

    def serialize(self, result: Any) -> str:
        for serializer in self.serializers:
            if serializer.can_handle(result):
                return serializer.serialize(result)
        return str(result)


_default_graph_operation_serializer = GraphOperationResultSerializerChain()


def serialize_graph_operation_result(result: Any) -> str:
    return _default_graph_operation_serializer.serialize(result)
