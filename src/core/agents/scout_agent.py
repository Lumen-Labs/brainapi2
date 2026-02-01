"""
File: /scout_agent.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List, Literal, Optional
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
from src.utils.tokens import token_detail_from_token_counts


class _ScoutEntity(BaseModel):
    """
    Scout entity.
    """

    type: str
    name: str
    properties: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = None
    polarity: Optional[Literal["positive", "negative", "neutral"]] = Field(
        default="neutral",
        description="The polarity of the entity.",
    )


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
        """
        Initialize a ScoutAgent with the provided adapters and reset internal agent and token-tracking state.

        Stores the provided LLM, cache, knowledge graph, vector store, and embeddings adapters on the instance, sets the agent and token_detail to None, and initializes input_tokens, output_tokens, cached_tokens, and reasoning_tokens counters to zero.
        """
        self.llm_adapter = llm_adapter
        self.cache_adapter = cache_adapter
        self.kg = kg
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.agent = None
        self.token_detail = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0

    def _get_tools(self, brain_id: str = "default") -> List[BaseTool]:
        """
        Provide the list of tools available to the agent for a given brain identifier.

        Parameters:
            brain_id (str): Identifier of the brain/context to retrieve tools for; defaults to "default".

        Returns:
            List[BaseTool]: A list of BaseTool instances available to the agent for the specified brain.
        """
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
        ingestion_session_id: Optional[str] = None,
    ) -> ScoutAgentResponse:
        """
        Extract entities from the provided text using the Scout agent and return a structured response containing the entities and token usage.

        Performs an LLM invocation (with optional targeting context), applies retries with exponential backoff on timeouts, enforces a per-invocation timeout, and accumulates token usage from the agent responses.

        Parameters:
            text: The input text to extract entities from.
            targeting: Optional Node providing contextual targeting information (name, description, properties) to bias extraction.
            brain_id: Identifier for the agent/brain configuration to use.
            timeout: Maximum seconds to wait for a single agent invocation before treating it as a timeout.
            max_retries: Maximum number of retry attempts for timed-out invocations using exponential backoff.

        Returns:
            A ScoutAgentResponse containing:
                - entities: list of extracted ScoutEntity objects.
                - input_tokens: accumulated input token count observed during the run.
                - output_tokens: accumulated output token count observed during the run.

        Raises:
            TimeoutError: If a single invocation exceeds `timeout`, or if all retry attempts fail due to timeouts.
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
                config={
                    "tags": ["scout_agent"],
                    "metadata": {
                        "agent": "scout_agent",
                        "brain_id": brain_id,
                        **(
                            {"ingestion_session_id": ingestion_session_id}
                            if ingestion_session_id
                            else {}
                        ),
                    },
                },
            )

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(TimeoutError),
            reraise=True,
        )
        def _invoke_agent_with_retry():
            """
            Invoke the agent in a separate thread, enforce the configured timeout, and update token accounting from the agent's response.

            Returns:
                dict: The agent response dictionary.

            Raises:
                TimeoutError: If the agent invocation exceeds the specified timeout.
            """
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_agent)
                    response = future.result(timeout=timeout)
                    for m in response.get("messages", []):
                        if hasattr(m, "usage_metadata"):
                            self._update_token_counts(m.usage_metadata)
                            self.token_detail = token_detail_from_token_counts(
                                self.input_tokens,
                                self.output_tokens,
                                self.cached_tokens,
                                self.reasoning_tokens,
                                "scout_agent",
                            )
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

        return ScoutAgentResponse(
            entities=[
                ScoutEntity(**entity.model_dump(mode="json"))
                for entity in response.get("structured_response", []).entities
            ],
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def _update_token_counts(self, usage_metadata: dict):
        """
        Update the agent's accumulated token counters from a usage metadata dictionary.

        Parameters:
            usage_metadata (dict): Metadata containing token counts. Expected keys:
                - "input_tokens": integer count to add to input_tokens (defaults to 0)
                - "output_tokens": integer count to add to output_tokens (defaults to 0)
                - "input_token_details": dict with optional "cache_read" integer to add to cached_tokens (defaults to 0)
                - "output_token_details": dict with optional "reasoning" integer to add to reasoning_tokens (defaults to 0)
        """
        # Base counts
        self.input_tokens += usage_metadata.get("input_tokens", 0)
        self.output_tokens += usage_metadata.get("output_tokens", 0)

        # Input details (caching)
        input_details = usage_metadata.get("input_token_details", {})
        self.cached_tokens += input_details.get("cache_read", 0)

        # Output details (reasoning)
        output_details = usage_metadata.get("output_token_details", {})
        self.reasoning_tokens += output_details.get("reasoning", 0)
