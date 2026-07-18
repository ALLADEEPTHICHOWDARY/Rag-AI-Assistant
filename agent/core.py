"""
agent/core.py
The Agent class ties together planning, routing, tool execution, memory,
and tracing into one loop: plan -> act -> reflect -> synthesize -> remember.
"""
import time
import uuid

from agent.planner import make_plan
from agent.router import choose_tool
from agent.memory import ConversationMemory, LongTermMemory
from agent.prompts import SYNTHESIS_PROMPT
from tools import ToolRegistry
from utils.logger import TraceLogger

MAX_STEPS_PER_SUBTASK = 3
FAILURE_PHRASES = ["i don't know", "not found", "no relevant", "failed", "error"]

# Tools whose output is already a complete, correct answer — routing these
# through an LLM synthesis step only risks a weak model garbling a fact
# that was already right (e.g. FLAN-T5 turning "25" into a sentence about
# what a calculator is). Skip synthesis entirely for these.
DIRECT_ANSWER_TOOLS = {"calculator"}


class Agent:
    def __init__(self, synthesize_fn=None, trace_dir: str = "traces"):
        """
        synthesize_fn: optional callable(prompt: str) -> str, wired up to a
        real LLM, used to write a final answer from gathered findings. If
        not provided, the agent returns the last successful tool result
        directly (see _default_synthesis) instead of building a text prompt
        and re-parsing it — that approach broke on multi-line tool results.
        """
        self.conversation = ConversationMemory()
        self.long_term = LongTermMemory()
        self.synthesize_fn = synthesize_fn
        self.logger = TraceLogger(trace_dir)

    def run(self, query: str) -> dict:
        trace_id = str(uuid.uuid4())[:8]
        started = time.time()
        self.conversation.add("user", query)

        plan = make_plan(query)
        findings = []  # list of (tool_name, result) — structured, not pre-joined text
        steps_log = [{"event": "plan_created", "plan": plan}]

        past = self.long_term.search(query)
        memory_note = "Relevant past exchanges:\n" + "\n".join(past) if past else "(none)"
        if past:
            steps_log.append({"event": "long_term_memory_hit", "results": past})

        for subtask in plan:
            used_tools = []
            for _ in range(MAX_STEPS_PER_SUBTASK):
                tool_name = choose_tool(subtask, used_tools)
                if tool_name is None:
                    break

                tool = ToolRegistry.get(tool_name)
                result = tool.fn(subtask)
                used_tools.append(tool_name)
                findings.append((tool_name, result))
                steps_log.append({
                    "event": "tool_call",
                    "subtask": subtask,
                    "tool": tool_name,
                    "result": result,
                })

                if not any(p in result.lower() for p in FAILURE_PHRASES):
                    break  # subtask answered, move to the next one

        should_use_llm = (
            self.synthesize_fn is not None
            and not (len(findings) == 1 and findings[0][0] in DIRECT_ANSWER_TOOLS)
        )

        if should_use_llm:
            findings_text = (
                "\n".join(f"[{t}] {r}" for t, r in findings)
                if findings else "(no tool results)"
            )
            prompt = SYNTHESIS_PROMPT.format(
                conversation=self.conversation.as_context(),
                memory=memory_note,
                query=query,
                findings=findings_text,
            )
            try:
                llm_answer = self.synthesize_fn(prompt)
            except Exception as e:
                llm_answer = None
                steps_log.append({"event": "synthesis_error", "error": str(e)})

            junk_outputs = {"none", "n/a", "unknown", "", "i don't know"}
            if llm_answer and llm_answer.strip().lower() not in junk_outputs:
                final_answer = llm_answer.strip()
            else:
                final_answer = self._default_synthesis(findings)
                steps_log.append({
                    "event": "synthesis_fallback",
                    "reason": f"LLM returned unusable output: {llm_answer!r}",
                })
        else:
            final_answer = self._default_synthesis(findings)

        self.conversation.add("agent", final_answer)
        self.long_term.save(query, final_answer)

        elapsed = round(time.time() - started, 2)
        steps_log.append({"event": "final_answer", "answer": final_answer, "elapsed_s": elapsed})
        self.logger.save(trace_id, steps_log)

        return {
            "answer": final_answer,
            "trace_id": trace_id,
            "steps": steps_log,
            "elapsed_s": elapsed,
        }

    @staticmethod
    def _default_synthesis(findings: list[tuple[str, str]]) -> str:
        """
        No LLM wired up: return the full result of the last successful
        tool call, unmodified — not a re-parsed fragment of it. If every
        tool failed, say so plainly instead of returning an error string.
        """
        if not findings:
            return "I couldn't find an answer to that."
        _, last_result = findings[-1]
        if any(p in last_result.lower() for p in FAILURE_PHRASES):
            return "I couldn't find an answer to that in the documents or the web."
        return last_result.strip()