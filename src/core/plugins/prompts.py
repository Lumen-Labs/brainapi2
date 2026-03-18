from threading import Lock
from typing import Optional


class PromptRegistry:
    _instance: Optional["PromptRegistry"] = None
    _lock: Lock = Lock()

    def __init__(self):
        self._overrides: dict[str, str] = {}
        self._extensions: dict[str, list[str]] = {}
        self._defaults: dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> "PromptRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_default(self, prompt_name: str, default_value: str) -> None:
        self._defaults[prompt_name] = default_value

    def override(self, prompt_name: str, new_prompt: str) -> None:
        self._overrides[prompt_name] = new_prompt

    def extend(self, prompt_name: str, suffix: str) -> None:
        self._extensions.setdefault(prompt_name, []).append(suffix)

    def get(self, prompt_name: str, default: Optional[str] = None) -> str:
        if prompt_name in self._overrides:
            return self._overrides[prompt_name]

        base = default if default is not None else self._defaults.get(prompt_name, "")

        extensions = self._extensions.get(prompt_name, [])
        if extensions:
            return base + "\n" + "\n".join(extensions)

        return base

    def reset(self, prompt_name: Optional[str] = None) -> None:
        if prompt_name:
            self._overrides.pop(prompt_name, None)
            self._extensions.pop(prompt_name, None)
        else:
            self._overrides.clear()
            self._extensions.clear()

    def list_overrides(self) -> list[str]:
        return list(self._overrides.keys())

    def list_extensions(self) -> list[str]:
        return list(self._extensions.keys())


prompt_registry = PromptRegistry.get_instance()
