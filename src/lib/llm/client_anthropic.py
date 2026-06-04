import json

from anthropic import Anthropic

from src.adapters.interfaces.llm import LLM
from src.config import config


class _AnthropicLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self.client = Anthropic(api_key=config.anthropic.api_key)
        self.langchain_model = None

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens or 1024,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in response.content if getattr(block, "text", None)]
        return "".join(parts)

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        response = None
        while max_retries > 0 and response is None:
            try:
                raw = self.generate_text(prompt, max_new_tokens=max_new_tokens)
                response = json.loads(raw.strip("```json").strip("```"))
            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    raise e
        return response


class AnthropicLLMClientSmall(_AnthropicLLMBase):
    def __init__(self):
        super().__init__(config.anthropic.small_llm_model)


class AnthropicLLMClientLarge(_AnthropicLLMBase):
    def __init__(self):
        super().__init__(config.anthropic.large_llm_model)


_llm_small_client_anthropic = AnthropicLLMClientSmall()
_llm_large_client_anthropic = AnthropicLLMClientLarge()
