# Agentic Upgrade

This turns the original single-pass RAG assistant into a proper agent:
it plans, chooses between tools, remembers past turns, and logs a full
reasoning trace for every run — instead of always doing one fixed
retrieve-then-answer pass.

## Architecture

```
User query
    │
    ▼
Planner ─── splits compound questions into ordered subtasks
    │
    ▼
Router ──── picks a tool per subtask (documents → calculator → web)
    │
    ▼
Tools ────── search_documents | calculator | web_search | summarize
    │
    ▼
Memory ───── short-term conversation buffer + persisted long-term Q&A
    │
    ▼
Synthesis ── combines findings into one final answer
    │
    ▼
Trace log ── every step written to /traces/<id>.json
```

## Why this is "agentic" and not just RAG

The original pipeline was: `query → retrieve → answer`. Fixed, one path,
no decisions. This version adds:

- **Planning** — a compound question like *"what's 12% of the Q3 revenue
  and does the contract mention termination fees"* gets split into two
  subtasks and handled separately.
- **Tool choice** — the agent decides per subtask whether it needs the
  documents, a calculator, or the web, rather than always doing the same
  thing.
- **Memory** — conversation context carries across turns, and past Q&A
  pairs are recalled if a similar question comes up again.
- **Observability** — every run is logged to `traces/<trace_id>.json` with
  the plan, every tool call, and the final answer. Open one to see
  exactly what the agent did and why.

## Files

| Path | Purpose |
|---|---|
| `tools/base.py` | Tool registry — tools self-register via decorator |
| `tools/document_search.py` | Wraps the existing FAISS/FLAN-T5 RAG pipeline |
| `tools/calculator.py` | Sandboxed arithmetic |
| `tools/web_search.py` | DuckDuckGo search, no API key |
| `tools/summarizer.py` | Compresses long tool output |
| `agent/planner.py` | Splits queries into subtasks |
| `agent/router.py` | Picks a tool per subtask (rule-based, LLM-based path included) |
| `agent/memory.py` | Short-term + long-term memory |
| `agent/core.py` | The main plan → act → reflect → synthesize loop |
| `utils/logger.py` | Writes JSON traces of every run |
| `app.py` | Streamlit UI with an expandable reasoning trace panel |
| `tests/` | pytest coverage for planner, router, and tools |

## Setup

```bash
pip install -r requirements.txt
pip install duckduckgo-search pytest   # for web_search tool + tests
```

In `tools/document_search.py`, change:
```python
from rag_core import get_answer
```
to match whatever your real RAG function is called.

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest tests/ -v
```

## Extending it further

- Swap `agent/router.py`'s rule-based `choose_tool()` for
  `choose_tool_llm()` to let a model make the routing decision instead.
- Swap `Agent(synthesize_fn=...)`'s default with a real LLM call so the
  final answer is generated, not just concatenated findings.
- Add a new tool: write a function, decorate it with
  `@ToolRegistry.register(name=..., description=...)`, import it in
  `tools/__init__.py`. No other file needs to change.
