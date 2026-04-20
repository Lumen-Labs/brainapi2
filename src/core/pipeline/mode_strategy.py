from abc import ABC, abstractmethod


class PipelineModeStrategy(ABC):
    mode: str

    @abstractmethod
    def should_extract_observations(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def scout_mode(self) -> str | None:
        raise NotImplementedError


class LightweightPipelineModeStrategy(PipelineModeStrategy):
    mode = "lightweight"

    def should_extract_observations(self) -> bool:
        return False

    def scout_mode(self) -> str | None:
        return "coarse"


class AccuratePipelineModeStrategy(PipelineModeStrategy):
    mode = "accurate"

    def should_extract_observations(self) -> bool:
        return True

    def scout_mode(self) -> str | None:
        return None


class PipelineModeStrategyFactory:
    def __init__(self):
        self._strategies = {
            "lightweight": LightweightPipelineModeStrategy(),
            "accurate": AccuratePipelineModeStrategy(),
        }

    def create(self, mode: str) -> PipelineModeStrategy:
        strategy = self._strategies.get(mode)
        if strategy is None:
            raise ValueError(f"Invalid PIPELINE_MODE: {mode}")
        return strategy


_pipeline_mode_strategy_factory = PipelineModeStrategyFactory()


def resolve_pipeline_mode_strategy(mode: str) -> PipelineModeStrategy:
    return _pipeline_mode_strategy_factory.create(mode)
