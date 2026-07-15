"""
agent/router.py
Decides which registered tool should handle a given subtask.
Rule-based by default (transparent and easy to debug); includes an
optional LLM-based path below for once you're comfortable with the loop.
"""
import re
from tools import ToolRegistry


def choose_tool(subtask: str, already_used: list[str]) -> str | None:
    """Return a tool name, or None if no tool applies (answer directly)."""

    if re.search(r"\d\s*[\+\-*/]\s*\d", subtask) and "calculator" not in already_used:
        return "calculator"

    if "search_documents" not in already_used:
        return "search_documents"

    if "web_search" not in already_used:
        return "web_search"

    return None


def choose_tool_llm(subtask: str, already_used: list[str], llm_fn) -> str | None:
    """
    Optional: ask an LLM to pick the tool instead of the rules above.
    `llm_fn` should be callable(prompt: str) -> str — e.g. wrapping your
    FLAN-T5 pipeline or an API-based chat model.
    """
    tool_menu = ToolRegistry.describe_all()
    prompt = (
        "You are a routing function for an AI agent. Given the tools below "
        "and a subtask, output ONLY the tool name to use next, or 'none' if "
        f"no tool is needed.\n\nTools:\n{tool_menu}\n\n"
        f"Already used this turn: {already_used}\n"
        f"Subtask: {subtask}\nTool name:"
    )
    choice = llm_fn(prompt).strip().lower()
    return choice if choice in ToolRegistry.all() else None
