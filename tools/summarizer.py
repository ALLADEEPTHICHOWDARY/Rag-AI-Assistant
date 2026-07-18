"""
tools/summarizer.py
Compresses long text (e.g. a big web search result or document chunk)
into a few sentences, using a local HuggingFace model — still no API key.
Model loads lazily so importing this module doesn't pay the load cost
until the tool is actually used.
"""
from tools.base import ToolRegistry

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


@ToolRegistry.register(
    name="summarize",
    description=(
        "Summarize a long piece of text into a few sentences. Use this when "
        "a tool result or document is too long to reason about directly."
    ),
)
def summarize(text: str) -> str:
    try:
        summarizer = _get_summarizer()
        word_count = len(text.split())
        if word_count < 40:
            return text  # too short to bother summarizing
        max_len = min(130, max(30, word_count // 2))
        result = summarizer(text[:4000], max_length=max_len, min_length=20, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        return f"[summarization failed: {e}]"
