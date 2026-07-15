"""
agent/memory.py
Two layers of memory:
  - ConversationMemory: short-term, in-process turn history for this session
  - LongTermMemory: persists past Q&A pairs to disk (JSON) so the agent can
    recall previous sessions without needing a separate database.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns = []  # list of {"role": "user"/"agent", "content": str}

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        self.turns = self.turns[-self.max_turns:]

    def as_context(self) -> str:
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)

    def clear(self):
        self.turns = []


class LongTermMemory:
    def __init__(self, path: str = "memory_store.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]")

    def save(self, query: str, answer: str):
        records = self._load()
        records.append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.path.write_text(json.dumps(records, indent=2))

    STOPWORDS = {
        "the", "is", "a", "an", "of", "to", "in", "on", "for", "and", "or",
        "what", "how", "when", "where", "who", "why", "does", "do", "did",
        "are", "was", "were", "be", "it", "this", "that", "with", "as",
        "at", "by", "from", "get", "gets",
    }

    def search(self, query: str, limit: int = 3) -> list[str]:
        """
        Keyword-overlap search over past Q&A, filtered to meaningful words
        only (stopwords excluded) and requiring at least half the query's
        meaningful words to match — otherwise "what is X?" ends up
        "relevant" to every other question ever asked.
        """
        records = self._load()
        query_words = {
            w for w in query.lower().split()
            if w not in self.STOPWORDS and len(w) > 2
        }
        if not query_words:
            return []

        scored = []
        for r in records:
            past_words = {
                w for w in r["query"].lower().split()
                if w not in self.STOPWORDS and len(w) > 2
            }
            overlap = len(query_words & past_words)
            if overlap >= 1 and overlap / len(query_words) >= 0.5:
                scored.append((overlap, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f"Q: {r['query']} / A: {r['answer']}" for _, r in scored[:limit]]

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []