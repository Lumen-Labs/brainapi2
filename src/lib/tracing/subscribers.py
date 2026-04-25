from abc import ABC, abstractmethod
from typing import Iterable

from src.lib.tracing.events import TraceEvent


class TraceSubscriber(ABC):
    @abstractmethod
    def handle(self, event: TraceEvent) -> None:
        raise NotImplementedError("handle method not implemented")

    def handle_many(self, events: Iterable[TraceEvent]) -> None:
        for event in events:
            self.handle(event)

