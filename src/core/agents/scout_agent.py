"""
File: /scout_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 2:27:56 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional
import uuid
from langchain.agents import create_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.cache import CacheAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.constants.kg import Node
from src.constants.prompts.scout_agent import (
    SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT,
    SCOUT_AGENT_SYSTEM_PROMPT,
)


class _ScoutEntity(BaseModel):
    """
    Scout entity.
    """

    type: str
    name: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None


class ScoutEntity(_ScoutEntity):
    """
    Scout entity.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class _ScoutAgentResponse(BaseModel):
    """
    Scout agent response containing the extracted entities.
    """

    entities: List[_ScoutEntity]


class ScoutAgentResponse(BaseModel):
    """
    Scout agent response containing the extracted entities (subjects and objects)
    from the text with their properties.
    """

    entities: List[ScoutEntity]
    input_tokens: int
    output_tokens: int


class ScoutAgent:
    """
    Scout agent.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache_adapter: CacheAdapter,
        kg: GraphAdapter,
        vector_store: VectorStoreAdapter,
        embeddings: EmbeddingsAdapter,
    ):
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None

    def _get_tools(self, brain_id: str = "default") -> List[BaseTool]:
        return []

    def _get_agent(
        self,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[BaseModel] = None,
        extra_system_prompt: Optional[dict] = None,
        brain_id: str = "default",
    ):
        system_prompt = SCOUT_AGENT_SYSTEM_PROMPT.format(
            extra_system_prompt=extra_system_prompt if extra_system_prompt else ""
        )

        self.agent = create_agent(
            model=self.llm_adapter.llm.langchain_model,
            tools=(tools if tools else self._get_tools(brain_id)),
            system_prompt=system_prompt,
            response_format=output_schema if output_schema else None,
            debug=os.environ.get("DEBUG", "false").lower() == "true",
        )

    def run(
        self,
        text: str,
        targeting: Optional[Node] = None,
        brain_id: str = "default",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> ScoutAgentResponse:
        """
        Run the scout agent.
        """

        self._get_agent(
            output_schema=_ScoutAgentResponse,
            brain_id=brain_id,
        )

        def _invoke_agent():
            return self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": SCOUT_AGENT_EXTRACT_ENTITIES_PROMPT.format(
                                text=text,
                                targeting=(
                                    f"""
                                The information is related to:
                                "{targeting.name}": {targeting.description}
                                {targeting.properties}
                                """
                                    if targeting
                                    else ""
                                ),
                            ),
                        }
                    ],
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry():
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent)
                    response = future.result(timeout=timeout)
                    return response
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Scout agent invoke timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )

        try:
            response = _invoke_agent_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"Scout agent invoke failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

        def _extract_tokens(r):
            response_metadata = r.get("response_metadata", {})
            token_usage = response_metadata.get("token_usage", {})
            if token_usage:
                return token_usage.get("prompt_tokens", 0), token_usage.get(
                    "completion_tokens", 0
                )
            usage_metadata = r.get("usage_metadata", {})
            if usage_metadata:
                return usage_metadata.get("input_tokens", 0), usage_metadata.get(
                    "output_tokens", 0
                )
            input_tokens = r.get("input_tokens", 0)
            output_tokens = r.get("output_tokens", 0)
            return input_tokens, output_tokens

        input_tokens, output_tokens = _extract_tokens(response)

        return ScoutAgentResponse(
            entities=[
                ScoutEntity(**entity.model_dump(mode="json"))
                for entity in response.get("structured_response", []).entities
            ],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
