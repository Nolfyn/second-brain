"""Тесты map-reduce синтеза: map идёт по секциям, а не по файлам."""

from brain.search import synthesiser


class FakeLLM:
    def __init__(self):
        self.map_calls = []  # нестриминговые вызовы (map)
        self.reduce_calls = 0  # стриминговые вызовы (reduce)

    def chat(self, messages, *, stream=False, **kwargs):
        if stream:
            self.reduce_calls += 1
            return iter([])
        self.map_calls.append(messages[-1]["content"])
        return "summary"


def _chunks():
    """25 чанков (>20 - порог map-reduce), два файла, по две секции."""
    out = []
    for i in range(25):
        fname = "A.docx" if i < 13 else "B.docx"
        section = ("Секция 1" if i < 7 else "Секция 2") if i < 13 else (
            "Секция 3" if i < 19 else "Секция 4"
        )
        out.append(
            {
                "file_path": f"C:/docs/{fname}",
                "text": f"Текст чанка {i}." * 50,
                "section": section,
                "source": "file",
                "score": None,
            }
        )
    return out


def test_map_runs_per_section_not_per_file(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(synthesiser, "get_llm", lambda: fake)

    list(synthesiser.synthesise("вопрос", _chunks()))

    assert len(fake.map_calls) == 4  # по числу секций
    assert fake.reduce_calls == 1
    # каждый map-вызов относится к одной секции одного файла
    for call in fake.map_calls:
        assert "Section:" in call
        assert "Question: вопрос" in call


def test_every_section_covered(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(synthesiser, "get_llm", lambda: fake)

    list(synthesiser.synthesise("вопрос", _chunks()))

    covered = "\n".join(fake.map_calls)
    for title in ("Секция 1", "Секция 2", "Секция 3", "Секция 4"):
        assert f"Section: {title}" in covered


def test_no_relevant_sections_yields_notice(monkeypatch):
    fake = FakeLLM()
    fake.chat = lambda messages, *, stream=False, **kw: (
        iter([]) if stream else "Nothing relevant found."
    )
    monkeypatch.setattr(synthesiser, "get_llm", lambda: fake)

    out = list(synthesiser.synthesise("вопрос", _chunks()))
    assert out == [
        "No relevant information was found in your documents for this query."
    ]


def test_oversized_section_split_into_batches(monkeypatch, tmp_path):
    """Переполненная секция делится на батчи - середина не теряется."""
    monkeypatch.setattr(synthesiser, "_SECTION_BATCH_CHARS", 400)
    fake = FakeLLM()
    monkeypatch.setattr(synthesiser, "get_llm", lambda: fake)

    # 25 чанков - больше порога map-reduce (>20)
    chunks = [
        {
            "file_path": "C:/docs/A.docx",
            "text": f"Часть {i}. " + "текст. " * 30,
            "section": "Большая секция",
            "source": "file",
            "score": None,
        }
        for i in range(25)
    ]
    list(synthesiser.synthesise("вопрос", chunks))

    assert len(fake.map_calls) == 25  # при бюджете 400 каждый чанк - свой батч
    assert "(часть 1)" in fake.map_calls[0]
    assert "(часть 25)" in fake.map_calls[-1]
    assert "[... пропуск середины секции ...]" not in fake.map_calls[0]
