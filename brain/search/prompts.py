"""Загрузка промптов из prompts.yaml с дефолтами в коде."""

from pathlib import Path

import yaml

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts.yaml"

_DEFAULTS = {
    "synthesis_system": (
        "You are a personal knowledge assistant. Answer using ONLY the "
        "document excerpts provided."
    ),
    "synthesis_user_tail": "Answer in {lang} ONLY.",
    "map_system": "Extract ONLY information literally present in the excerpts.",
    "reduce_system": "Combine the summaries into a comprehensive answer.",
    "expansion": 'Generate 3 short search queries about:\n"{query}"',
}

_cache: dict | None = None


def get_prompts(force_reload: bool = False) -> dict:
    """Промпты из YAML поверх дефолтов. Результат кэшируется."""
    global _cache
    if _cache is not None and not force_reload:
        return _cache
    prompts = dict(_DEFAULTS)
    if _PROMPTS_PATH.exists():
        data = yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8")) or {}
        prompts.update({k: v for k, v in data.items() if isinstance(v, str)})
    _cache = prompts
    return prompts


def render(name: str, **kwargs) -> str:
    return get_prompts()[name].format(**kwargs)
