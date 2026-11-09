"""Тесты абстракции LLM: фабрика провайдеров, сборка payload, стриминг."""

from unittest.mock import MagicMock, patch

import pytest

from brain.llm import (
    AnthropicLLM,
    OllamaLLM,
    OpenAILLM,
    _iter_sse_json,
    get_llm,
)


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = get_llm()
    assert isinstance(llm, OllamaLLM)


def test_factory_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    llm = get_llm()
    assert isinstance(llm, AnthropicLLM)
    assert llm.model


def test_factory_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    llm = get_llm()
    assert isinstance(llm, OpenAILLM)


def test_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonexistent")
    with pytest.raises(ValueError):
        get_llm()


@patch("brain.llm.ollama.chat")
def test_ollama_chat_nonstream(mock_chat):
    mock_chat.return_value = {"message": {"content": "привет"}}
    llm = OllamaLLM(model="qwen2.5:7b")
    out = llm.chat([{"role": "user", "content": "вопрос"}])
    assert out == "привет"
    assert mock_chat.call_args.kwargs["model"] == "qwen2.5:7b"


@patch("brain.llm.ollama.chat")
def test_ollama_stream_yields_strings(mock_chat):
    mock_chat.return_value = iter(
        [
            {"message": {"content": "раз"}},
            {"message": {"content": " два"}},
        ]
    )
    llm = OllamaLLM(model="m")
    tokens = list(llm.chat([{"role": "user", "content": "q"}], stream=True))
    assert tokens == ["раз", " два"]


@patch("brain.llm.httpx.Client")
def test_anthropic_system_extracted_from_messages(mock_client_cls):
    mock_client = MagicMock()
    mock_client.post.return_value.json.return_value = {
        "content": [{"type": "text", "text": "ответ"}]
    }
    mock_client_cls.return_value = mock_client

    llm = AnthropicLLM(model="claude-test", api_key="k")
    out = llm.chat(
        [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "вопрос"},
        ]
    )
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["system"] == "система"
    assert payload["messages"] == [{"role": "user", "content": "вопрос"}]
    assert payload["max_tokens"] > 0
    assert out == "ответ"


@patch("brain.llm.httpx.Client")
def test_openai_payload_keeps_system_inline(mock_client_cls):
    mock_client = MagicMock()
    mock_client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    mock_client_cls.return_value = mock_client

    llm = OpenAILLM(model="kimi-test", api_key="k", base_url="https://api.example.com/v1")
    out = llm.chat(
        [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "вопрос"},
        ]
    )
    url = mock_client.post.call_args.args[0]
    payload = mock_client.post.call_args.kwargs["json"]
    assert url == "https://api.example.com/v1/chat/completions"
    assert payload["messages"][0] == {"role": "system", "content": "система"}
    assert out == "ok"


def test_sse_parsing_stops_at_done():
    lines = [
        'data: {"a": 1}',
        "",
        "data: [DONE]",
        'data: {"a": 2}',
    ]
    assert list(_iter_sse_json(iter(lines))) == [{"a": 1}]


def test_anthropic_text_delta_extraction():
    from brain.llm import _anthropic_text

    chunk = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}}
    assert _anthropic_text(chunk) == "x"


def test_openai_delta_extraction():
    from brain.llm import _openai_text

    chunk = {"choices": [{"delta": {"content": "y"}}]}
    assert _openai_text(chunk) == "y"
    assert _openai_text({"choices": [{"delta": {}}]}) == ""
