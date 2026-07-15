"""
utils/logger.py
Writes a JSON trace of every agent run to disk. This is the kind of
observability that matters in real agent systems: open any trace file
and see exactly what the agent planned, called, and returned, in order.
"""
import json
from pathlib import Path


class TraceLogger:
    def __init__(self, trace_dir: str = "traces"):
        self.dir = Path(trace_dir)
        self.dir.mkdir(exist_ok=True)

    def save(self, trace_id: str, steps: list[dict]) -> Path:
        path = self.dir / f"{trace_id}.json"
        path.write_text(json.dumps(steps, indent=2))
        return path
