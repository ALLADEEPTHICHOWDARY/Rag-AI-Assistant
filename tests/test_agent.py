import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner import make_plan
from agent.router import choose_tool


def test_planner_splits_compound_query():
    plan = make_plan("What is 2 + 2 and then summarize the report")
    assert len(plan) == 2


def test_planner_keeps_simple_query_whole():
    plan = make_plan("What does the report say about revenue?")
    assert len(plan) == 1


def test_router_picks_calculator_for_math():
    assert choose_tool("12 * 4", already_used=[]) == "calculator"


def test_router_does_not_repeat_a_used_tool():
    assert choose_tool("12 * 4", already_used=["calculator"]) != "calculator"


def test_router_falls_back_to_documents_before_web():
    tool = choose_tool("what is the refund policy?", already_used=[])
    assert tool == "search_documents"
