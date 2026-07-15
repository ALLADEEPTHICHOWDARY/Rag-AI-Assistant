"""
agent/planner.py
Breaks a user request into an ordered list of subtasks before execution.
Most questions will just be a single subtask, but this layer is what
turns a "one-shot tool call" into genuine multi-step planning — the
thing that separates an agent from a plain router.
"""
import re


def make_plan(query: str) -> list[str]:
    """
    Split a query into subtasks. Handles the common "compound question"
    pattern (joined by 'and then', 'and also', or ';') and otherwise
    returns the query as a single-step plan. Kept conservative on purpose:
    over-splitting hurts more than under-splitting.
    """
    parts = re.split(r"\s+and then\s+|\s+and also\s+|;\s*", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [query]
