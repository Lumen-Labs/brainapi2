from abc import ABC, abstractmethod
from typing import Iterable

from src.lib.tracing.events import TraceEvent


class TraceSubscriber(ABC):
    @abstractmethod
    def publish(self, event: TraceEvent) -> None:
        raise NotImplementedError("publish method not implemented")

    def publish_many(self, events: Iterable[TraceEvent]) -> None:
        for event in events:
            self.publish(event)

