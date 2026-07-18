import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.calculator import calculator


def test_calculator_basic():
    assert calculator("2 + 2") == "4"


def test_calculator_handles_parentheses():
    assert calculator("(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_input():
    result = calculator("__import__('os').system('ls')")
    assert "error" in result.lower()
