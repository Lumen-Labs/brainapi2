"""
File: /client_small.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

os.environ["GRPC_DNS_RESOLVER"] = "native"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.adapters.interfaces.llm import LLM
from src.config import config

_parent_pid = os.getpid()


class LLMClientSmall(LLM):
    """
    Large language model client, used for main agents and other high-level tasks.
    """

    def __init__(self):
        self._client = None
        self._model = None
        self._langchain_model = None
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self.default_timeout = 120

    def _check_fork(self):
        if os.getpid() != self._pid:
            self._client = None
            self._langchain_model = None
            self._pid = os.getpid()

    @property
    def client(self):
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from google import genai

                    self._client = genai.Client(
                        vertexai=True,
                        project=config.gcp.project_id,
                        location="global",
                        http_options={"api_version": "v1"},
                    )
        return self._client

    @property
    def model(self):
        if self._model is None:
            self._model = config.gcp.small_llm_model
        return self._model

    @property
    def langchain_model(self):
        if self._langchain_model is None:
            with self._lock:
                if self._langchain_model is None:
                    from langchain_google_vertexai import ChatVertexAI

                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                        config.gcp.credentials_path
                    )
                    self._langchain_model = ChatVertexAI(
                        model_name=self.model,
                        project=config.gcp.project_id,
                        max_retries=5,
                        location="global",
                        request_timeout=120,
                        streaming=False,
                        wait_exponential_kwargs={
                            "multiplier": 1,
                            "min": 2,
                            "max": 60,
                        },
                    )
        return self._langchain_model

    def generate_text(
        self, prompt: str, max_new_tokens: int = 100000, timeout: int = None
    ) -> str:
        if not prompt or len(prompt.strip()) == 0:
            return "Input prompt is empty"

        timeout = timeout or self.default_timeout
        from google.genai.types import GenerateContentConfig

        def _generate():
            return self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=GenerateContentConfig(
                    response_mime_type="text/plain",
                    **({"max_output_tokens": max_new_tokens} if max_new_tokens else {}),
                ),
            )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((TimeoutError, Exception)),
            reraise=True,
        )
        def _generate_with_retry():
            executor = ThreadPoolExecutor(max_workers=1)
            future = None
            try:
                future = executor.submit(_generate)
                response = future.result(timeout=timeout)
                return response.text
            except FutureTimeoutError:
                if future:
                    future.cancel()
                raise TimeoutError(
                    f"LLM generate_text timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        try:
            return _generate_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"LLM generate_text failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except TimeoutError:
            raise

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = 100000,
        max_retries: int = 3,
        timeout: int = None,
    ) -> dict:
        timeout = timeout or self.default_timeout
        from google.genai.types import GenerateContentConfig

        def _generate():
            _response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    **({"max_output_tokens": max_new_tokens} if max_new_tokens else {}),
                ),
            )
            _response = _response.text.strip("").strip("```")
            return json.loads(_response)

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (TimeoutError, json.JSONDecodeError, Exception)
            ),
            reraise=True,
        )
        def _generate_with_retry():
            executor = ThreadPoolExecutor(max_workers=1)
            future = None
            try:
                future = executor.submit(_generate)
                response = future.result(timeout=timeout)
                return response
            except FutureTimeoutError:
                if future:
                    future.cancel()
                raise TimeoutError(
                    f"LLM generate_json timed out after {timeout} seconds. "
                    "This may indicate a network issue or the LLM service is unresponsive."
                )
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Failed to parse JSON: {e}", e.doc, e.pos)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        try:
            return _generate_with_retry()
        except RetryError as e:
            last_attempt = e.last_attempt
            raise TimeoutError(
                f"LLM generate_json failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            ) from last_attempt.exception()
        except (TimeoutError, json.JSONDecodeError):
            raise


_llm_small_client = LLMClientSmall()
