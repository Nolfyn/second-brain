"""Тесты API чат-сессий: создание, вопрос, история, удаление."""

import json

import pytest
from fastapi.testclient import TestClient

import brain.api.main as api
from brain.chat_store import ChatStore


class FakeLLM:
    """Стримит фиксированный ответ, запоминает сообщения."""

    def __init__(self):
        self.seen_messages = []

    def chat(self, messages, *, stream=False, **kwargs):
        if not stream:
            return "summary"
        self.seen_messages.append(messages)
        return iter(["ок"])


class FakeResult:
    file_path = "C:/docs/Notes.docx"
    text = "текст"
    score = 0.5
    chunk_index = 0
    source = "semantic"
    section = None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        api, "_chat_store", ChatStore(str(tmp_path / "chat.db"))
    )
    monkeypatch.setattr(
        api._retriever, "retrieve", lambda query: [FakeResult()]
    )
    monkeypatch.setattr(
        "brain.search.synthesiser.get_llm", lambda: FakeLLM()
    )
    return TestClient(api.app)


def _stream_events(response):
    events = []
    for line in response.text.splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def test_create_and_list_sessions(client):
    r1 = client.post("/sessions")
    r2 = client.post("/sessions")
    assert r1.status_code == 200

    sessions = client.get("/sessions").json()
    assert [s["id"] for s in sessions] == [r2.json()["id"], r1.json()["id"]]


def test_ask_saves_exchange_with_sources(client):
    sid = client.post("/sessions").json()["id"]

    response = client.post(f"/sessions/{sid}/ask", json={"query": "вопрос"})
    assert response.status_code == 200
    events = _stream_events(response)
    assert events[0]["type"] == "sources"
    assert events[1] == {"type": "token", "content": "ок"}

    session = client.get(f"/sessions/{sid}").json()
    assert session["title"] == "вопрос"
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"]
    assert session["messages"][1]["content"] == "ок"
    assert session["messages"][1]["sources"][0]["file_path"] == "C:/docs/Notes.docx"


def test_ask_includes_prior_history(client, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("brain.search.synthesiser.get_llm", lambda: fake)
    sid = client.post("/sessions").json()["id"]

    client.post(f"/sessions/{sid}/ask", json={"query": "первый вопрос"})
    client.post(f"/sessions/{sid}/ask", json={"query": "уточнение"})

    # стриминговый вызов (reduce) получил историю: первый вопрос + ответ
    reduce_messages = fake.seen_messages[-1]
    roles = [m["role"] for m in reduce_messages]
    assert roles.count("user") == 2  # история + текущий вопрос
    contents = "\n".join(m["content"] for m in reduce_messages)
    assert "первый вопрос" in contents
    assert "уточнение" in contents


def test_session_not_found(client):
    assert client.post("/sessions/nope/ask", json={"query": "q"}).status_code == 404
    assert client.get("/sessions/nope").status_code == 404


def test_delete_session(client):
    sid = client.post("/sessions").json()["id"]
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.get("/sessions").json() == []
