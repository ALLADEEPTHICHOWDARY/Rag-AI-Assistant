"""
tools/document_search.py
Wraps the existing FAISS + FLAN-T5 RAG pipeline as a tool the agent can
call. Uses the real rag_core.py functions: build_vectorstore(document)
and get_response(query, vectorstore).

Since get_response() needs a vectorstore object (built in app.py once a
file is uploaded), this module holds a reference to it that app.py sets
after "Build Knowledge Base" is clicked.
"""
from tools.base import ToolRegistry
from rag_core import get_response

_vectorstore = None


def set_vectorstore(vectorstore) -> None:
    """Call this from app.py right after build_vectorstore() succeeds."""
    global _vectorstore
    _vectorstore = vectorstore


@ToolRegistry.register(
    name="search_documents",
    description=(
        "Search the user's uploaded documents (PDF/DOCX/TXT) for an answer. "
        "Use this FIRST for any question that could be about the uploaded content."
    ),
)
def search_documents(query: str) -> str:
    if _vectorstore is None:
        return "[document search failed: no knowledge base built yet]"
    try:
        return get_response(query, _vectorstore)
    except Exception as e:
        return f"[document search failed: {e}]"
