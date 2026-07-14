"""
CONCEPT: Persistence and checkpointing — attaching a `checkpointer` to a
compiled graph so its FULL STATE survives between separate `.invoke()`
calls, keyed by a `thread_id`.

../../LangChain/memory/memory.py's HONESTY NOTE names this file directly: LangChain's
`RunnableWithMessageHistory` is deprecated in favor of exactly this
mechanism. The upgrade is bigger than it first looks:
`RunnableWithMessageHistory` persists ONE thing — a list of chat
messages. A LangGraph checkpointer persists the ENTIRE state object —
here, a running list of notes AND whatever other fields a graph's state
happens to have — because it's a property of the GRAPH's execution, not a
message-history wrapper bolted onto a chain. Same `thread_id`-keyed
isolation as `memory.py`'s `session_id` (see the test in this directory's
README for proof two threads never see each other's notes), but now
anything in the state, not just messages, comes back automatically.

HONESTY NOTE: `InMemorySaver`, used below, is still in-process only —
kill this script and the notes are gone, same limitation
../../LangChain/memory/memory.py was upfront about for its own in-memory
store. LangGraph ships persistent checkpointer backends (Postgres, SQLite)
for surviving a real restart; swapping `InMemorySaver()` for one of those
is the only change `build_graph` below would need — nothing about the
graph shape or node code changes. Contrast with
../../Task_and_State_Management/checkpointing/checkpointing.py, which
gets real cross-process durability today, at the cost of writing and
parsing its own JSON checkpoint file by hand.

Use case: a running notes list per conversation thread — add a note, come
back later (same thread_id) and the list is still there; switch thread_id
and it's empty again. Type 'exit' to end the session.
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class NotesState(TypedDict):
    new_note: str
    # CONCEPT: only `new_note` needs to be passed into each `.invoke()`
    # call — `notes` is never re-supplied by the caller after the first
    # call. The checkpointer remembers its last value for this thread_id
    # and the operator.add reducer appends this call's contribution to it
    # (same reducer mechanic as ../state_graph_basics/state_graph_basics.py's `log` field).
    notes: Annotated[list[str], operator.add]


def add_note(state: NotesState) -> dict:
    return {"notes": [state["new_note"]]}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(NotesState)
    graph.add_node("add_note", add_note)
    graph.add_edge(START, "add_note")
    graph.add_edge("add_note", END)

    # CONCEPT: this one argument is the entire feature — compiling WITH a
    # checkpointer means every `.invoke()` against a given thread_id reads
    # that thread's last saved state before running, and writes the new
    # state back after. Compiling without one (every other template in
    # this repo so far) means each `.invoke()` starts from scratch.
    return graph.compile(checkpointer=InMemorySaver())


def add(graph: CompiledStateGraph, thread_id: str, note: str) -> list[str]:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"new_note": note}, config=config)
    return result["notes"]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script "
            "(kept for consistency with the rest of this repo — this template's "
            "graph makes no model calls of its own).",
            file=sys.stderr,
        )
        sys.exit(1)

    graph = build_graph()

    print("Per-thread notes graph (persistence/checkpointing demo). Type 'exit' to quit.")
    print("Each turn starts with a thread id — reuse one to keep notes, switch to isolate them.\n")
    print("Try: thread 'work' -> \"Finish the report\", then thread 'personal' -> \"Buy milk\"\n")

    while True:
        thread_id = input("Thread id (or 'exit'): ").strip()
        if thread_id.lower() == "exit":
            print("Goodbye!")
            break
        if not thread_id:
            continue
        note = input(f"[{thread_id}] New note: ").strip()
        if not note:
            continue

        notes = add(graph, thread_id, note)
        print(f"[{thread_id}] All notes so far: {notes}\n")


if __name__ == "__main__":
    main()
