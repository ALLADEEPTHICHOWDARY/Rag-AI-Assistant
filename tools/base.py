"""
tools/base.py
Every tool registers itself here via a decorator. This means the agent
never hardcodes a tool list — it discovers what's available at runtime,
so adding a new capability is just: write a function, decorate it, done.
"""
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]


class ToolRegistry:
    _tools: Dict[str, Tool] = {}

    @classmethod
    def register(cls, name: str, description: str):
        def decorator(fn: Callable[[str], str]):
            cls._tools[name] = Tool(name=name, description=description, fn=fn)
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Tool:
        return cls._tools[name]

    @classmethod
    def all(cls) -> Dict[str, Tool]:
        return cls._tools

    @classmethod
    def describe_all(cls) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in cls._tools.values())
