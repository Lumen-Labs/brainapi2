import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable

logger = logging.getLogger(__name__)


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

    def _serialize_record(self, record: Any) -> dict | str:
        record_data = getattr(record, "data", None)
        if callable(record_data):
            return record_data()
        try:
            return dict(record)
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to map neo4j record, using string fallback: %s", exc)
            return str(record)

    def _extract_keys(self, result: Any) -> list | None:
        keys_accessor = getattr(result, "keys", None)
        if callable(keys_accessor):
            try:
                return list(keys_accessor())
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Failed to extract neo4j result keys from callable: %s", exc
                )
                return None
        if isinstance(keys_accessor, Iterable) and not isinstance(
            keys_accessor, (str, bytes)
        ):
            return list(keys_accessor)
        return None

    def serialize(self, result: Any) -> str:
        records = result.records or []
        limited_records = records[:20]
        serialized_records = []
        for record in limited_records:
            serialized_records.append(self._serialize_record(record))
        keys = self._extract_keys(result)
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
