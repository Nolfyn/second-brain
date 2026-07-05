from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the web via DuckDuckGo (no API key needed).
    Returns a list of {"url": ..., "snippet": ...} dicts.
    Returns an empty list on failure rather than raising —
    web search is an optional enhancement, not core functionality.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    return [
        {"url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw_results
    ]
