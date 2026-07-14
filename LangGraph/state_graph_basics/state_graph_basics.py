"""
CONCEPT: StateGraph basics — a graph of named nodes, wired together with
explicit edges, that all read from and write to one shared state object.

../../LangChain/langgraph_workflows/langgraph_workflows.py is a first taste of this from inside the
LangChain topic — a 3-node graph with one conditional branch. This
template stays INSIDE the LangGraph topic and goes one level deeper into
the primitives that file used without dwelling on: the graph literally
starts at the special `START` node and ends at the special `END` node
(langgraph_workflows.py uses the `set_entry_point()` convenience wrapper,
which is really just `add_edge(START, key)` under the hood — this
template uses `add_edge(START, ...)` directly to make that explicit), and
state doesn't just get overwritten node to node — with the right type
annotation, LangGraph can be told how to COMBINE what a node returns with
what's already there, rather than replacing it outright.

That combining behavior is `log`'s `Annotated[list[str], operator.add]`
type below: every node returns just the ONE new log line it wants to add
(`{"log": ["..."]}`, not the whole growing list), and LangGraph
concatenates it onto the existing list automatically using the reducer
function named in the annotation. Compare this to
../../Memory/working_memory/working_memory.py's scratchpad, where the
PROGRAM (a plain dict a Python function mutates) is what accumulates
state across steps — here the GRAPH FRAMEWORK does the accumulating,
driven by a type annotation, not by any node's code explicitly appending.

Use case: a linear 3-step order fulfillment pipeline — validate, charge,
draft a confirmation message — where each step appends one line to a
shared audit log that's fully assembled by the time the graph reaches
END. Type 'exit' to end the session.
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512

# A tiny mock inventory — a stand-in for a real product database, same
# role as basic_agentic_tools.py's TASKS_FILE.
_INVENTORY = {"widget": 12, "gadget": 0, "gizmo": 5}


class OrderState(TypedDict):
    order_id: str
    item: str
    # CONCEPT: Annotated[..., operator.add] — the reducer. Without this
    # annotation, a node returning {"log": [...]} would REPLACE the
    # existing log; with it, LangGraph calls operator.add(old, new) —
    # i.e. list concatenation — to merge them instead.
    log: Annotated[list[str], operator.add]
    confirmation: str


def validate_order(state: OrderState) -> dict:
    in_stock = _INVENTORY.get(state["item"], 0) > 0
    if not in_stock:
        return {"log": [f"validate_order: '{state['item']}' is OUT OF STOCK"]}
    return {"log": [f"validate_order: '{state['item']}' is in stock ({_INVENTORY[state['item']]} left)"]}


def charge_payment(state: OrderState) -> dict:
    # A real node would call a payment API; this one just simulates success.
    return {"log": [f"charge_payment: charged order {state['order_id']}"]}


def draft_confirmation(state: OrderState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(
        f"Write one short, friendly sentence confirming order {state['order_id']} "
        f"for a '{state['item']}' has shipped."
    )
    return {"log": ["draft_confirmation: wrote customer confirmation"], "confirmation": reply.content}


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(OrderState)
    graph.add_node("validate_order", validate_order)
    graph.add_node("charge_payment", charge_payment)
    graph.add_node("draft_confirmation", lambda state: draft_confirmation(state, llm))

    # CONCEPT: START and END are singleton markers (not real nodes you
    # write code for) that mark where a run begins and where it's allowed
    # to finish. Every graph needs at least one edge INTO it from START
    # and at least one edge OUT of some node TO END.
    graph.add_edge(START, "validate_order")
    graph.add_edge("validate_order", "charge_payment")
    graph.add_edge("charge_payment", "draft_confirmation")
    graph.add_edge("draft_confirmation", END)

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    graph = build_graph(llm)

    print("Order fulfillment graph (StateGraph basics demo). Type 'exit' to quit.\n")
    print("Try: order_id='A100', item='widget'\n")

    while True:
        order_id = input("Order id (or 'exit'): ").strip()
        if order_id.lower() == "exit":
            print("Goodbye!")
            break
        item = input("Item: ").strip()
        if not item:
            continue

        result = graph.invoke({"order_id": order_id, "item": item, "log": [], "confirmation": ""})
        print("\n[audit log]")
        for line in result["log"]:
            print(f"  - {line}")
        print(f"\n{result['confirmation']}\n")


if __name__ == "__main__":
    main()
