"""
tools/calculator.py
A sandboxed arithmetic evaluator. No API key, no external calls.
"""
import re
from tools.base import ToolRegistry


@ToolRegistry.register(
    name="calculator",
    description=(
        "Evaluate a math expression, e.g. '12 * (3 + 4)'. Use this for "
        "arithmetic instead of trying to compute it yourself."
    ),
)
def calculator(text: str) -> str:
    # Pull the arithmetic expression out of a natural-language subtask,
    # e.g. "what is 5 + 7" -> "5 + 7", rather than requiring the whole
    # input to already be pure math.
    # Grab the longest contiguous run of math-safe characters that
    # actually contains a digit (so we don't match stray punctuation).
    candidates = re.findall(r"[0-9+\-*/(). ]+", text)
    candidates = [c.strip() for c in candidates if re.search(r"\d", c)]
    if not candidates:
        return "[calculator error: no arithmetic expression found]"
    expression = max(candidates, key=len)
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"[calculator error: {e}]"
