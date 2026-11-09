"""Абстракция над провайдерами LLM.

Провайдер выбирается переменной окружения LLM_PROVIDER:
  - ollama (по умолчанию) - локальная модель, настройка через OLLAMA_MODEL;
  - anthropic - Claude API, ключ ANTHROPIC_API_KEY, модель ANTHROPIC_MODEL;
  - openai - любой OpenAI-совместимый endpoint, ключ OPENAI_API_KEY,
  модель OPENAI_MODEL, адрес OPENAI_BASE_URL.
"""

import json
import os
from collections.abc import Iterator

import httpx
import ollama

DEFAULT_NUM_PREDICT = 2048


def get_llm() -> "BaseLLM":
    """Создает клиента LLM по переменным окружения."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    if provider == "anthropic":
        return AnthropicLLM(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            api_key=_require_env("ANTHROPIC_API_KEY"),
        )
    if provider == "openai":
        return OpenAILLM(
            model=os.getenv("OPENAI_MODEL", "kimi-k2-0905-preview"),
            api_key=_require_env("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.moonshot.ai/v1"),
        )
    raise ValueError(f"Неизвестный провайдер LLM: '{provider}'")


def _require_env(name: str) -> str:
    """Значение переменной окружения или понятная ошибка."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Переменная окружения {name} не задана - "
            "проверьте .env (см. .env.example)"
        )
    return value


class BaseLLM:
    model: str

    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | Iterator[str]:
        """Без stream=True возвращает полный текст, иначе - генератор токенов."""
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    def __init__(self, model: str):
        self.model = model

    def chat(self, messages, *, stream=False, temperature=None, max_tokens=None):
        options = {"num_predict": max_tokens or DEFAULT_NUM_PREDICT}
        gpu = os.getenv("OLLAMA_NUM_GPU_LAYERS")
        if gpu:
            options["num_gpu"] = int(gpu)
        if temperature is not None:
            options["temperature"] = temperature
        response = ollama.chat(
            model=self.model, messages=messages, options=options, stream=stream
        )
        if not stream:
            return response["message"]["content"]
        return (c["message"]["content"] for c in response if c["message"]["content"])


class AnthropicLLM(BaseLLM):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = httpx.Client(
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=180.0,
        )

    def chat(self, messages, *, stream=False, temperature=None, max_tokens=None):
        system, msgs = _split_system(messages)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or DEFAULT_NUM_PREDICT,
            "messages": msgs,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if not stream:
            resp = self._client.post(self.URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []))
        return self._stream({**payload, "stream": True})

    def _stream(self, payload):
        with self._client.stream("POST", self.URL, json=payload) as resp:
            for chunk in _iter_sse_json(resp.iter_lines()):
                text = _anthropic_text(chunk)
                if text:
                    yield text


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        )

    def chat(self, messages, *, stream=False, temperature=None, max_tokens=None):
        payload = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if not stream:
            resp = self._client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return self._stream({**payload, "stream": True})

    def _stream(self, payload):
        with self._client.stream("POST", self._url, json=payload) as resp:
            for chunk in _iter_sse_json(resp.iter_lines()):
                text = _openai_text(chunk)
                if text:
                    yield text


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic требует system отдельным полем, а не ролью в messages."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


def _iter_sse_json(lines) -> Iterator[dict]:
    """Парсит SSE-стрим: json-payload из data:-строк до маркера [DONE]."""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def _anthropic_text(chunk: dict) -> str:
    if chunk.get("type") == "content_block_delta":
        return chunk.get("delta", {}).get("text", "")
    return ""


def _openai_text(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("delta", {}).get("content", "") or ""
