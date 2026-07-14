"""
CONCEPT: Multi-agent subgraphs — composing a fully compiled `StateGraph`
AS A NODE inside a larger graph, so a specialist's internal multi-step
behavior stays real graph structure (inspectable, independently testable,
its own nodes and edges) rather than being flattened into a single
function call.

../../Multi_Agent_Systems/orchestrator/orchestrator.py's orchestrator sees each specialist as a
TOOL: `delegate_to_researcher(...)` is one opaque call that goes out,
runs an entire separate agent, and comes back with a single result —
whatever that specialist did internally is invisible from the
orchestrator's side, and ../../Multi_Agent_Systems/worker_agent/ is where you'd go to see
inside one. Here, there's no such boundary: `billing_subgraph` and
`technical_subgraph` below are each a real, multi-node `StateGraph`
compiled on their own, and the top-level `router_graph` adds the
COMPILED GRAPH OBJECT itself as a node with `add_node("billing",
billing_subgraph)` — not a function that calls it internally. Both the
outer graph and the specialists share the same `TicketState` shape, so a
specialist's nodes read and write the SAME state object the outer graph
does, with no translation layer at the boundary.

HONESTY NOTE: ../state_graph_basics/state_graph_basics.py's `Annotated[list[str],
operator.add]` reducer looks like the obvious way to accumulate a shared
`steps_taken` log here too — but tested against a REAL compiled subgraph
boundary, it double-applies whatever the parent wrote immediately before
handing off to the subgraph (verified directly: a 2-node parent -> 1-node
subgraph came back with the parent's two log entries duplicated,
`['a1-ran', 'a2-ran', 'a1-ran', 'a2-ran', 'b-ran']`, not the expected five
entries once each). That's a genuine LangGraph subgraph/reducer
interaction, not a hypothetical — so `steps_taken` below is accumulated
MANUALLY (`state["steps_taken"] + [...]`, plain overwrite semantics, no
`Annotated` reducer) instead. It's more verbose per node, but correct
across the parent/subgraph boundary, which the reducer version silently
was not.

Use case: a support ticket router — `classify` picks a category, then a
conditional edge hands off to whichever specialist subgraph matches
(`billing`: verify_account -> resolve_billing; `technical`: diagnose ->
resolve_technical), each specialist appending its own steps to a shared
`steps_taken` log so the full trace — across both the outer graph and
whichever specialist ran — is visible in one place at the end. Type
'exit' to end the session.
"""

from __future__ import annotations

import os
import sys
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512


class TicketState(TypedDict):
    ticket_text: str
    category: str
    # CONCEPT: plain list, no Annotated reducer — see the module
    # docstring's HONESTY NOTE for why an operator.add reducer is unsafe
    # across a parent/subgraph boundary. Every node below accumulates by
    # explicitly reading and re-writing the whole list.
    steps_taken: list[str]
    resolution: str


# ---------------------------------------------------------------------------
# SPECIALIST 1: billing subgraph — two internal steps of its own.
# ---------------------------------------------------------------------------
def verify_account(state: TicketState) -> dict:
    return {"steps_taken": state["steps_taken"] + ["billing: verified account status"]}


def resolve_billing(state: TicketState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"As a billing specialist, resolve this ticket in one sentence:\n\n{state['ticket_text']}")
    return {"steps_taken": state["steps_taken"] + ["billing: drafted resolution"], "resolution": reply.content}


def build_billing_subgraph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(TicketState)
    graph.add_node("verify_account", verify_account)
    graph.add_node("resolve_billing", lambda state: resolve_billing(state, llm))
    graph.add_edge(START, "verify_account")
    graph.add_edge("verify_account", "resolve_billing")
    graph.add_edge("resolve_billing", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# SPECIALIST 2: technical subgraph — a different two internal steps.
# ---------------------------------------------------------------------------
def diagnose(state: TicketState) -> dict:
    return {"steps_taken": state["steps_taken"] + ["technical: ran diagnostics"]}


def resolve_technical(state: TicketState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"As a technical specialist, resolve this ticket in one sentence:\n\n{state['ticket_text']}")
    return {"steps_taken": state["steps_taken"] + ["technical: drafted resolution"], "resolution": reply.content}


def build_technical_subgraph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(TicketState)
    graph.add_node("diagnose", diagnose)
    graph.add_node("resolve_technical", lambda state: resolve_technical(state, llm))
    graph.add_edge(START, "diagnose")
    graph.add_edge("diagnose", "resolve_technical")
    graph.add_edge("resolve_technical", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# TOP-LEVEL router graph
# ---------------------------------------------------------------------------
def classify(state: TicketState) -> dict:
    text = state["ticket_text"].lower()
    category = "billing" if any(w in text for w in ("bill", "charge", "refund", "invoice")) else "technical"
    return {"category": category, "steps_taken": state["steps_taken"] + [f"router: classified as {category}"]}


def route_after_classify(state: TicketState) -> Literal["billing", "technical"]:
    return state["category"]


def build_router_graph(llm: BaseChatModel) -> CompiledStateGraph:
    billing_subgraph = build_billing_subgraph(llm)
    technical_subgraph = build_technical_subgraph(llm)

    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    # CONCEPT: the compiled subgraphs are the nodes themselves — LangGraph
    # invokes them exactly like it would invoke a plain function node,
    # because a compiled StateGraph is itself invokable the same way.
    graph.add_node("billing", billing_subgraph)
    graph.add_node("technical", technical_subgraph)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"billing": "billing", "technical": "technical"})
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    router_graph = build_router_graph(llm)

    print("Support ticket router (multi-agent subgraphs demo). Type 'exit' to quit.\n")
    print("Try: \"I was charged twice for my subscription this month\"\n")

    while True:
        ticket_text = input("Ticket text: ").strip()
        if ticket_text.lower() == "exit":
            print("Goodbye!")
            break
        if not ticket_text:
            continue

        result = router_graph.invoke({"ticket_text": ticket_text, "category": "", "steps_taken": [], "resolution": ""})
        print(f"  [category: {result['category']}]")
        for step in result["steps_taken"]:
            print(f"  [step] {step}")
        print(f"\n{result['resolution']}\n")


if __name__ == "__main__":
    main()
