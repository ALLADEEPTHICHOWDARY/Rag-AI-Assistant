"""
tools/__init__.py
Importing this package runs every tool module below, which registers
each tool into ToolRegistry via the @ToolRegistry.register decorator.
"""
from tools.base import ToolRegistry
from tools import document_search, calculator, web_search, summarizer  # noqa: F401

__all__ = ["ToolRegistry"]
