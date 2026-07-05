"""
Tests for the synthesiser's prompt construction.

These exercise `_build_user_prompt`, the function that assembles the final
prompt sent to the LLM from retrieved chunks and optional web results.
"""

from brain.search.synthesiser import _build_user_prompt


def test_build_user_prompt_with_local_chunks_only():
    chunks = [
        {"file_path": "notes.docx", "text": "Some content about stoicism."}
    ]
    prompt = _build_user_prompt(
        "what is stoicism?", chunks, web_results=None, lang="English"
    )
    assert "stoicism" in prompt.lower()
    assert "notes.docx" in prompt
    assert "WEB SEARCH RESULTS" not in prompt


def test_build_user_prompt_with_web_results():
    chunks = [{"file_path": "notes.docx", "text": "Some content."}]
    web = [{"url": "https://example.com", "snippet": "Some web content."}]
    prompt = _build_user_prompt(
        "a question", chunks, web_results=web, lang="English"
    )
    assert "WEB SEARCH RESULTS" in prompt
    assert "example.com" in prompt


def test_build_user_prompt_with_no_results():
    prompt = _build_user_prompt(
        "a question", [], web_results=None, lang="English"
    )
    assert "No relevant content was found" in prompt
