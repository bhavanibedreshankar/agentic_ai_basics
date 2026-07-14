"""
CONCEPT: Callbacks and Tracing — observing every step of a chain (which
model calls happened, with what input, how long each took) from OUTSIDE
the chain's own code, by attaching a handler object rather than editing
the chain's logic.

../../Safety_and_Control/audit_trail/audit_trail.py builds the same kind of record — a
structured, append-only JSONL log of everything an agent did — but by
hand: `log.record(...)` calls are written directly into `run_turn()` at
each point something happens. That means logging is welded to the
business logic; adding a second thing to log, or logging a completely
different chain, means editing that function again. A LangChain
`BaseCallbackHandler` inverts this: you implement `on_chain_start`,
`on_chat_model_start`, `on_llm_end`, etc. ONCE, then attach the same
handler instance to ANY chain via `config={"callbacks": [...]}` — the
chain's own code (see ../chains/chains.py's `build_review_chain`) never
mentions logging at all. This is what "tracing" means in the LangChain
ecosystem: LangSmith (a hosted service, not used here — see the module
docstring's honesty note below) is built entirely on this same callback
mechanism, just with a handler that ships events to a server instead of a
local file.

HONESTY NOTE: LangChain's own hosted tracing product, LangSmith, would
normally be the "real" way to do this — set `LANGCHAIN_TRACING_V2=true`
and every run shows up in a web UI automatically. That requires an
external account and API key this environment doesn't have, so this
template builds a local, file-based tracer using the exact same
`BaseCallbackHandler` interface LangSmith's own integration uses — the
mechanism demonstrated here is the real one, just writing to a JSONL file
instead of shipping to a hosted backend.

Use case: a two-step draft-then-critique pipeline, traced end to end; the
`print_trace` call at the end of each turn replays the JSONL log from disk,
same proof-of-independence idea as audit_trail.py's `print_audit_log`.
Type 'exit' to end the conversation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512

TRACE_LOG_PATH = Path(__file__).parent / "trace_log.jsonl"

DRAFT_PROMPT = ChatPromptTemplate.from_messages([("human", "Write one sentence answering: {question}")])
CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [("human", "Critique this answer in one short sentence — is it accurate and complete?\n\n{draft}")]
)


class TracingHandler(BaseCallbackHandler):
    """CONCEPT: the tracer itself. Every method here corresponds to a
    specific lifecycle event LangChain fires as a chain runs — implementing
    only the ones you care about is fine, the rest default to no-ops.
    `run_id` uniquely identifies one step's execution, which is how
    on_chat_model_start's start time gets matched back up with the right
    on_llm_end call, even when multiple model calls are in flight (e.g.
    ../chains/chains.py's RunnableParallel branches running concurrently).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.session_id = uuid.uuid4().hex[:8]
        self._start_times: dict[UUID, float] = {}

    def _record(self, event_type: str, payload: dict) -> None:
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        name = (serialized or {}).get("name", "chain")
        print(f"  [trace] chain_start: {name}")
        self._record("chain_start", {"chain": name, "inputs": {k: str(v)[:80] for k, v in inputs.items()}})

    def on_chat_model_start(self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, **kwargs: Any) -> None:
        self._start_times[run_id] = time.monotonic()
        preview = messages[0][-1].content[:80] if messages and messages[0] else ""
        print(f"  [trace] llm_start: {preview!r}")
        self._record("llm_start", {"prompt_preview": preview})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed_ms = round((time.monotonic() - self._start_times.pop(run_id, time.monotonic())) * 1000, 1)
        text = response.generations[0][0].text
        print(f"  [trace] llm_end: {elapsed_ms}ms -> {text[:80]!r}")
        self._record("llm_end", {"elapsed_ms": elapsed_ms, "output_preview": text[:80]})


def print_trace(path: Path, session_id: str) -> None:
    # CONCEPT: proving the trace is a genuine independent record, same as
    # audit_trail.py's print_audit_log — reads only from disk, never from
    # in-memory handler state.
    if not path.exists():
        print("(no trace log yet)")
        return
    print(f"\n=== Trace for session {session_id} ===")
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["session_id"] != session_id:
                continue
            detail = {k: v for k, v in entry.items() if k not in ("session_id", "timestamp", "event_type")}
            print(f"  {entry['timestamp']}  {entry['event_type']}: {detail}")


def build_draft_and_critique(llm: BaseChatModel) -> tuple[Runnable, Runnable]:
    return DRAFT_PROMPT | llm, CRITIQUE_PROMPT | llm


def run_traced(draft_chain: Runnable, critique_chain: Runnable, question: str, handler: TracingHandler) -> tuple[str, str]:
    # CONCEPT: attaching the tracer — `config={"callbacks": [...]}` is how
    # ANY Runnable's `.invoke()` accepts a handler, with zero changes to
    # DRAFT_PROMPT, CRITIQUE_PROMPT, or the chains built from them.
    config = {"callbacks": [handler]}
    draft = draft_chain.invoke({"question": question}, config=config).content
    critique = critique_chain.invoke({"draft": draft}, config=config).content
    return draft, critique


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    draft_chain, critique_chain = build_draft_and_critique(llm)
    handler = TracingHandler(TRACE_LOG_PATH)

    print(f"Draft + critique pipeline (callbacks/tracing demo) — session {handler.session_id}.")
    print("Type 'exit' to end the conversation.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print_trace(TRACE_LOG_PATH, handler.session_id)
            print("\nGoodbye!")
            break
        if not question:
            continue

        draft, critique = run_traced(draft_chain, critique_chain, question, handler)
        print(f"\nDraft: {draft}\nCritique: {critique}\n")


if __name__ == "__main__":
    main()
