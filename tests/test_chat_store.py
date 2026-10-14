"""Тесты хранилища чат-сессий."""

from brain.chat_store import ChatStore


def test_create_and_list_sessions(tmp_path):
    store = ChatStore(str(tmp_path / "t.db"))
    s1 = store.create_session()
    s2 = store.create_session()

    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == [s2, s1]  # новые первыми
    assert sessions[0]["title"] == "Новый чат"


def test_add_message_sets_title_from_first_question(tmp_path):
    store = ChatStore(str(tmp_path / "t.db"))
    sid = store.create_session()

    store.add_message(sid, "user", "Чем занимался в 2017?")
    store.add_message(sid, "assistant", "Ответ.", sources=[{"file_path": "a.docx"}])

    session = store.get_session(sid)
    assert session["title"] == "Чем занимался в 2017?"
    assert len(session["messages"]) == 2
    assert session["messages"][1]["sources"] == [{"file_path": "a.docx"}]


def test_get_session_with_history(tmp_path):
    store = ChatStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.add_message(sid, "user", "вопрос 1")
    store.add_message(sid, "assistant", "ответ 1")
    store.add_message(sid, "user", "уточнение")

    history = store.get_history(sid)
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[-1]["content"] == "уточнение"


def test_delete_session_removes_messages(tmp_path):
    store = ChatStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.add_message(sid, "user", "вопрос")

    store.delete_session(sid)
    assert store.get_session(sid) is None
    assert store.list_sessions() == []


def test_history_limit(tmp_path):
    store = ChatStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    for i in range(30):
        store.add_message(sid, "user", f"вопрос {i}")
        store.add_message(sid, "assistant", f"ответ {i}")

    history = store.get_history(sid, limit=10)
    assert len(history) == 10
    assert history[-1]["content"] == "ответ 29"
