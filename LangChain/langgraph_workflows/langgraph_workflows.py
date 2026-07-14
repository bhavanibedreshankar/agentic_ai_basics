"""
CONCEPT: LangGraph Workflows — a lightweight first look at building an
explicit graph of steps, where the EDGES between steps (not just the
step logic) can branch, and in general graphs can cycle back on
themselves. This is a brief taste from inside the LangChain topic; the
full ../../LangGraph/ topic directory goes much deeper (state
persistence, human-in-the-loop interrupts, streaming, multi-agent
subgraphs) — start there once this makes sense.

../chains/chains.py's `RunnableBranch` already does conditional routing —
so what does a graph add? `RunnableBranch` picks between two runnables and
the result is final; the shape of the pipeline is still a straight line
(a DAG) fixed at compose time. A `StateGraph` is a first-class graph of
named NODES connected by named EDGES, including CONDITIONAL edges whose
destination is computed at runtime from the current state — and, unlike
LCEL's `|` pipelines, those edges are free to point back at a node that
already ran, which is what lets a LangGraph agent loop (ask model -> call
tool -> ask model again) run an unbounded number of times. This template
doesn't use a cycle — see ../../LangGraph/conditional_routing/ for one —
but the branching state-graph shape is the same mechanism the deeper
LangGraph topic builds on.

Use case: the same ticket-triage idea as ../prompt_templates/prompt_templates.py, restructured as a
three-node graph: `classify` always runs first, then a conditional edge
routes to either `auto_respond` (routine tickets) or `escalate` (urgent
ones) based on the classification — never both, and the graph ends either
way. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512


# CONCEPT: graph state — a TypedDict every node reads from and writes back
# to. Unlike a chain's data flowing linearly through `|`, ALL nodes in a
# graph share this one evolving state object; each node returns only the
# keys it wants to update.
class TicketState(TypedDict):
    ticket_text: str
    priority: str
    response: str


def classify(state: TicketState, llm: BaseChatModel) -> dict:
    # CONCEPT: a node is just a function: (state) -> partial state update.
    # This one calls the model directly rather than through an LCEL chain,
    # to keep this intro focused on the graph shape — nothing stops a node
    # from being a full chain internally (../../LangGraph/ shows that).
    reply = llm.invoke(
        f"Classify this support ticket's priority as exactly one word — "
        f"'routine' or 'urgent':\n\n{state['ticket_text']}"
    )
    return {"priority": reply.content.strip().lower()}


def auto_respond(state: TicketState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"Write a brief, friendly acknowledgment reply to this routine ticket:\n\n{state['ticket_text']}")
    return {"response": reply.content}


def escalate(state: TicketState, llm: BaseChatModel) -> dict:
    return {"response": f"[escalated to a human agent] {state['ticket_text']}"}


def route_after_classify(state: TicketState) -> Literal["auto_respond", "escalate"]:
    # CONCEPT: the conditional edge — a plain function of the current
    # state returning the NAME of the next node. This is what makes
    # branching a property of the graph's edges, not of a Runnable's
    # internal logic like RunnableBranch's condition callables.
    return "escalate" if state["priority"] == "urgent" else "auto_respond"


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(TicketState)
    graph.add_node("classify", lambda state: classify(state, llm))
    graph.add_node("auto_respond", lambda state: auto_respond(state, llm))
    graph.add_node("escalate", lambda state: escalate(state, llm))

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_after_classify, {"auto_respond": "auto_respond", "escalate": "escalate"})
    graph.add_edge("auto_respond", END)
    graph.add_edge("escalate", END)

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    graph = build_graph(llm)

    print("Ticket triage graph (LangGraph intro). Type 'exit' to quit.\n")
    print("Try: \"The entire production site is down for all customers\"\n")

    while True:
        ticket_text = input("Ticket text: ").strip()
        if ticket_text.lower() == "exit":
            print("Goodbye!")
            break
        if not ticket_text:
            continue

        result = graph.invoke({"ticket_text": ticket_text, "priority": "", "response": ""})
        print(f"  [priority: {result['priority']}]")
        print(f"\n{result['response']}\n")


if __name__ == "__main__":
    main()
