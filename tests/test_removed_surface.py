"""Удалённый функционал: агентный режим, web-поиск, MCP не должны остаться в проекте."""

import inspect


def test_search_package_does_not_export_agent_or_web_search():
    import brain.search

    assert not hasattr(brain.search, "Agent")
    assert not hasattr(brain.search, "web_search")


def test_cli_has_no_ask_agent_command():
    from brain.cli import app

    command_names = [c.name for c in app.registered_commands]
    assert "ask-agent" not in command_names


def test_api_has_no_agent_endpoint():
    from brain.api.main import app

    routes = [getattr(r, "path", "") for r in app.routes]
    assert "/agent" not in routes


def test_synthesise_has_no_web_results_param():
    from brain.search.synthesiser import synthesise

    params = inspect.signature(synthesise).parameters
    assert "web_results" not in params
