"""
tools/web_search.py
Web search fallback for when the documents don't have the answer.
Primary: ddgs (the current name for what used to be duckduckgo_search —
that package is deprecated and renamed). Fallback: Wikipedia's public
search API, which is far more reliable since it's not a scraper.
Neither requires an API key.
"""
import re
from tools.base import ToolRegistry


def _search_ddgs(query: str) -> str | None:
    from ddgs import DDGS
    results = list(DDGS().text(query, max_results=3))
    if not results:
        return None
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)


def _search_wikipedia(query: str) -> str | None:
    import requests
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        },
        timeout=8,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    if not hits:
        return None
    cleaned = [re.sub(r"<[^<]+?>", "", h["snippet"]) for h in hits]
    return "\n".join(f"- {h['title']}: {c}" for h, c in zip(hits, cleaned))


@ToolRegistry.register(
    name="web_search",
    description=(
        "Search the public web. Use this only if the documents did not "
        "contain the answer, or the question is clearly about current "
        "events / general knowledge outside the uploaded files."
    ),
)
def web_search(query: str) -> str:
    errors = []

    try:
        result = _search_ddgs(query)
        if result:
            return result
        errors.append("ddgs: no results (likely rate-limited)")
    except Exception as e:
        errors.append(f"ddgs: {e}")

    try:
        result = _search_wikipedia(query)
        if result:
            return result
        errors.append("wikipedia: no results")
    except Exception as e:
        errors.append(f"wikipedia: {e}")

    return f"[web search failed — {'; '.join(errors)}]"