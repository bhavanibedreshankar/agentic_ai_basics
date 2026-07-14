"""
CONCEPT: Human-in-the-loop via `interrupt()` — pausing a graph mid-run at
a specific node, handing control to a human, and resuming exactly where
it left off once they respond, as a first-class graph primitive rather
than a blocking function call.

../../Execution_Loops/human_in_the_loop/human_in_the_loop.py does approval gating by hand: a
plain Python `input()` call sits directly in the middle of the agent's
loop, and the whole process just blocks on stdin until someone types a
response. That works, but the "paused" state is nothing more than the
Python interpreter sitting on a call stack — kill the process while it's
waiting and the pending approval is gone. `interrupt()` here does
something categorically different: calling it inside a node raises a
special signal that unwinds the graph run entirely and returns control to
the caller, with the interrupt's payload attached to the result under
`"__interrupt__"`. The graph isn't blocked on anything — it's stopped,
with its state (which node it was in, everything computed so far) saved
by the checkpointer from ../persistence_and_checkpointing/. Resuming later
is a SEPARATE `.invoke(Command(resume=...))` call, which could happen
seconds or days afterward, from an entirely different process, as long as
it's pointed at the same `thread_id` and a checkpointer that actually
persisted (this template's `InMemorySaver` doesn't survive a restart —
see ../persistence_and_checkpointing/'s honesty note for the same
limitation and what fixes it).

Use case: the expense-approval scenario from
../../LangChain/agents_and_tools/agents_and_tools.py's tool set, but now
anything over a threshold pauses for a human decision instead of an agent
freely calling `approve_expense`. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

APPROVAL_THRESHOLD = 100.0


class ExpenseState(TypedDict):
    description: str
    amount: float
    decision: str
    result: str


def request_approval(state: ExpenseState) -> dict:
    if state["amount"] > APPROVAL_THRESHOLD:
        # CONCEPT: interrupt() pauses the graph HERE, mid-node, and
        # surfaces `value` to whoever is watching this thread. The graph
        # doesn't resume until a later, separate call passes
        # Command(resume=...) against the SAME thread_id — at which point
        # `interrupt()` returns that resume value and this line continues
        # as if it had been a normal function call all along.
        decision = interrupt({"description": state["description"], "amount": state["amount"]})
    else:
        decision = "approved"  # below threshold: no human needed at all
    return {"decision": decision}


def finalize(state: ExpenseState) -> dict:
    verb = "Approved" if state["decision"] == "approved" else "Denied"
    return {"result": f"{verb}: {state['description']} (${state['amount']:.2f})"}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(ExpenseState)
    graph.add_node("request_approval", request_approval)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "request_approval")
    graph.add_edge("request_approval", "finalize")
    graph.add_edge("finalize", END)
    # A checkpointer is REQUIRED for interrupt() to work at all — the
    # paused state has to be saved somewhere for the later resume call to
    # find. See ../persistence_and_checkpointing/ for what this argument does on its own.
    return graph.compile(checkpointer=InMemorySaver())


def submit_expense(graph: CompiledStateGraph, thread_id: str, description: str, amount: float) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"description": description, "amount": amount, "decision": "", "result": ""}, config=config)


def resume_with_decision(graph: CompiledStateGraph, thread_id: str, decision: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=decision), config=config)


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

    print(f"Expense approval graph (human-in-the-loop demo). Threshold: ${APPROVAL_THRESHOLD:.2f}.")
    print("Type 'exit' to end the session.\n")
    print("Try: \"New laptop\" for $1500 — watch it pause for approval.\n")

    thread_counter = 0
    while True:
        description = input("Expense description (or 'exit'): ").strip()
        if description.lower() == "exit":
            print("Goodbye!")
            break
        try:
            amount = float(input("Amount: $").strip())
        except ValueError:
            print("Please enter a number.\n")
            continue

        thread_counter += 1
        thread_id = f"expense-{thread_counter}"
        result = submit_expense(graph, thread_id, description, amount)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n  [paused for approval] {payload['description']} — ${payload['amount']:.2f}")
            human_decision = input("  Approve? (yes/no): ").strip().lower()
            decision = "approved" if human_decision.startswith("y") else "denied"
            result = resume_with_decision(graph, thread_id, decision)

        print(f"\n{result['result']}\n")


if __name__ == "__main__":
    main()
