import json
from abc import ABC, abstractmethod

from src.constants.tasks.ingestion import IngestionTaskJsonArgs, IngestionTaskTextArgs


class IngestionTextStrategy(ABC):
    @abstractmethod
    def can_handle(self, payload_data: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract(self, payload_data: object) -> str:
        raise NotImplementedError


class RawTextIngestionStrategy(IngestionTextStrategy):
    def can_handle(self, payload_data: object) -> bool:
        return isinstance(payload_data, IngestionTaskTextArgs)

    def extract(self, payload_data: object) -> str:
        if not isinstance(payload_data, IngestionTaskTextArgs):
            raise ValueError("Invalid payload type for text ingestion strategy")
        return payload_data.text_data


class JsonIngestionStrategy(IngestionTextStrategy):
    def can_handle(self, payload_data: object) -> bool:
        return isinstance(payload_data, IngestionTaskJsonArgs)

    def extract(self, payload_data: object) -> str:
        if not isinstance(payload_data, IngestionTaskJsonArgs):
            raise ValueError("Invalid payload type for json ingestion strategy")
        return json.dumps(payload_data.json_data)


class IngestionTextStrategyFactory:
    def __init__(self, strategies: list[IngestionTextStrategy] | None = None):
        self._strategies = strategies or [
            RawTextIngestionStrategy(),
            JsonIngestionStrategy(),
        ]

    def create(self, payload_data: object) -> IngestionTextStrategy:
        for strategy in self._strategies:
            if strategy.can_handle(payload_data):
                return strategy
        raise ValueError(
            f"Unsupported ingestion payload type: {type(payload_data).__name__}"
        )


_default_ingestion_text_strategy_factory = IngestionTextStrategyFactory()


def extract_ingestion_text(payload_data: object) -> str:
    strategy = _default_ingestion_text_strategy_factory.create(payload_data)
    return strategy.extract(payload_data)
