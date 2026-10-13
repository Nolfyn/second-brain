"""Тесты конфигурации промптов: дефолты, YAML-переопределение, подстановка."""

from brain.search import prompts


def test_default_prompts_loaded():
    p = prompts.get_prompts()
    for key in (
        "synthesis_system",
        "synthesis_user_tail",
        "map_system",
        "reduce_system",
        "expansion",
    ):
        assert key in p
        assert p[key].strip()


def test_yaml_overrides_defaults(tmp_path, monkeypatch):
    yaml_file = tmp_path / "prompts.yaml"
    yaml_file.write_text("synthesis_system: КАСТОМНЫЙ ПРОМПТ\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_PROMPTS_PATH", yaml_file)

    p = prompts.get_prompts(force_reload=True)
    assert p["synthesis_system"] == "КАСТОМНЫЙ ПРОМПТ"
    # непереопределённые ключи остаются дефолтными
    assert p["expansion"].strip()


def test_render_substitutes_placeholders():
    text = prompts.render("expansion", query="тестовый запрос")
    assert "тестовый запрос" in text
