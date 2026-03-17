from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient
from openai import OpenAI

from superni_benchmark.config import GenerationConfig, ModelConfig, PromptConfig


@dataclass(slots=True)
class ModelResponse:
    text: str
    raw: dict[str, Any]
    attempts: int
    latency_seconds: float


class BaseModelClient:
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        prompt_config: PromptConfig,
    ) -> None:
        self.model_config = model_config
        self.generation_config = generation_config
        self.prompt_config = prompt_config

    def generate(self, prompt: str) -> ModelResponse:
        raise NotImplementedError


class OpenAIModelClient(BaseModelClient):
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        prompt_config: PromptConfig,
    ) -> None:
        super().__init__(model_config, generation_config, prompt_config)
        api_key = os.environ[model_config.api_key_env or "OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key, timeout=generation_config.timeout_seconds)

    def generate(self, prompt: str) -> ModelResponse:
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                payload = {
                    "model": self.model_config.model,
                    "instructions": self.prompt_config.system_prompt,
                    "input": prompt,
                    "max_output_tokens": self.generation_config.max_output_tokens,
                }
                reasoning_effort = self._reasoning_effort()
                if reasoning_effort:
                    payload["reasoning"] = {"effort": reasoning_effort}
                response = self.client.responses.create(**payload)
                latency_seconds = time.perf_counter() - started
                return ModelResponse(
                    text=(response.output_text or "").strip(),
                    raw=response.model_dump(mode="json"),
                    attempts=attempt,
                    latency_seconds=latency_seconds,
                )
            except Exception:
                if attempt >= self.generation_config.max_retries:
                    raise
                time.sleep(2**attempt)

    def _reasoning_effort(self) -> str | None:
        return self.model_config.reasoning_effort or self.generation_config.reasoning_effort


class HuggingFaceChatClient(BaseModelClient):
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        prompt_config: PromptConfig,
    ) -> None:
        super().__init__(model_config, generation_config, prompt_config)
        token = os.environ[model_config.api_key_env or "HF_TOKEN"]
        self.client = InferenceClient(
            provider=model_config.provider or "hyperbolic",
            api_key=token,
            timeout=generation_config.timeout_seconds,
        )

    def generate(self, prompt: str) -> ModelResponse:
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                completion = self.client.chat_completion(
                    model=self.model_config.model,
                    messages=[
                        {"role": "system", "content": self.prompt_config.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.generation_config.temperature,
                    max_tokens=self.generation_config.max_output_tokens,
                    extra_body=self.model_config.extra_body or None,
                )
                latency_seconds = time.perf_counter() - started
                content = completion.choices[0].message.content or ""
                return ModelResponse(
                    text=content.strip(),
                    raw=_serialize_provider_response(completion),
                    attempts=attempt,
                    latency_seconds=latency_seconds,
                )
            except Exception:
                if attempt >= self.generation_config.max_retries:
                    raise
                time.sleep(2**attempt)


def _serialize_provider_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        payload = response.dict()
        if isinstance(payload, dict):
            return payload
    if isinstance(response, dict):
        return response
    if hasattr(response, "__dict__"):
        return {key: _to_jsonable(value) for key, value in vars(response).items()}
    return {"value": _to_jsonable(response)}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {key: _to_jsonable(item) for key, item in vars(value).items()}
    return repr(value)


def build_model_client(
    model_config: ModelConfig,
    generation_config: GenerationConfig,
    prompt_config: PromptConfig,
) -> BaseModelClient:
    if model_config.backend == "openai":
        return OpenAIModelClient(model_config, generation_config, prompt_config)
    if model_config.backend == "huggingface-chat":
        return HuggingFaceChatClient(model_config, generation_config, prompt_config)
    raise ValueError(f"Unsupported backend: {model_config.backend}")
