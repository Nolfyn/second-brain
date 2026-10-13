"""Тесты сборки промпта для синтеза."""

from brain.search.synthesiser import _build_user_prompt


def test_prompt_with_chunks():
    chunks = [
        {"file_path": "notes.docx", "text": "Some content about stoicism."}
    ]
    prompt = _build_user_prompt("what is stoicism?", chunks, lang="English")
    assert "stoicism" in prompt.lower()
    assert "notes.docx" in prompt


def test_prompt_with_no_results():
    prompt = _build_user_prompt("a question", [], lang="English")
    assert "No relevant content was found" in prompt
