"""
CONCEPT: Conditional routing — an edge whose destination is computed at
runtime from the current state, via `add_conditional_edges`. Branching
alone isn't new (../../LangChain/chains/chains.py's `RunnableBranch` and
../../LangChain/langgraph_workflows/langgraph_workflows.py already pick between two fixed
outcomes) — what's new here is a CYCLE: a conditional edge that can route
back to a node that already ran, something no LCEL chain (a fixed,
linear/parallel DAG built at compose time) can express at all. That's the
capability a graph adds over a chain: the number of times a step runs
isn't decided when the pipeline is built, it's decided while it's running.

Two raw-SDK templates build the pieces of this same idea by hand:
../../Execution_Loops/max_iterations/max_iterations.py caps a loop at a
fixed number of turns so it can't run forever, and
../../Execution_Loops/interrupts_breakpoints/interrupts_breakpoints.py
stops a loop early once a condition is met. This template's cycle does
both at once, as graph structure instead of a `while` loop: `check_draft`
routes back to `write_draft` while the draft still fails validation, and
also routes to `END` once `MAX_ATTEMPTS` is hit, exactly mirroring
max_iterations.py's safety cap — but here it's an edge destination
decided by `route_after_check`, not a counter checked inside a Python
`while`.

Use case: keep revising a one-paragraph product pitch until it passes a
plain-code validator (at least 3 sentences, mentions the product name),
capped at 3 attempts so a genuinely stuck pipeline can't loop forever.
Type 'exit' to end the session.
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
MAX_ATTEMPTS = 3


class PitchState(TypedDict):
    product: str
    draft: str
    feedback: str
    attempts: int


def write_draft(state: PitchState, llm: BaseChatModel) -> dict:
    if state["feedback"]:
        # CONCEPT: this is what makes it a real revision loop, not just a
        # retry — each pass through write_draft sees the PREVIOUS attempt's
        # feedback and is asked to fix it, the same "next step sees prior
        # step's output" idea ../../LangChain/chains/chains.py's `|`
        # composition uses, just looped instead of linear.
        prompt = (
            f"Revise this product pitch for '{state['product']}' to fix the "
            f"problem noted, in 3-4 sentences:\n\nDraft: {state['draft']}\n"
            f"Problem: {state['feedback']}"
        )
    else:
        prompt = f"Write a short product pitch (3-4 sentences) for '{state['product']}'."

    reply = llm.invoke(prompt)
    return {"draft": reply.content, "attempts": state["attempts"] + 1}


def check_draft(state: PitchState) -> dict:
    # CONCEPT: a plain-code gate, same spirit as
    # ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's validate_outline — cheap,
    # deterministic, no model call.
    sentence_count = state["draft"].count(".") + state["draft"].count("!")
    if sentence_count < 3:
        return {"feedback": f"Too short — only {sentence_count} sentence(s), need at least 3."}
    if state["product"].lower() not in state["draft"].lower():
        return {"feedback": f"Doesn't mention the product name '{state['product']}'."}
    return {"feedback": ""}


def route_after_check(state: PitchState) -> Literal["write_draft", "end"]:
    # CONCEPT: the conditional edge function — a plain function of state
    # returning where to go next. Passing ("write_draft" or "end")
    # instead of a boolean is what lets this same shape extend to more
    # than two destinations if a graph needs it.
    if not state["feedback"]:
        return "end"  # passed validation
    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"  # safety cap reached — stop even though it never passed
    return "write_draft"  # loop back and try again


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(PitchState)
    graph.add_node("write_draft", lambda state: write_draft(state, llm))
    graph.add_node("check_draft", check_draft)

    graph.add_edge(START, "write_draft")
    graph.add_edge("write_draft", "check_draft")
    # CONCEPT: the cycle. "write_draft" appears both as a node ABOVE and
    # as a possible destination here, from a node ("check_draft") that
    # runs strictly after it — this is the edge that turns the graph into
    # a loop rather than a straight line.
    graph.add_conditional_edges("check_draft", route_after_check, {"write_draft": "write_draft", "end": END})

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    graph = build_graph(llm)

    print("Pitch refinement graph (conditional routing / cycles demo). Type 'exit' to quit.\n")
    print("Try: product='NoiseAway headphones'\n")

    while True:
        product = input("Product (or 'exit'): ").strip()
        if product.lower() == "exit":
            print("Goodbye!")
            break
        if not product:
            continue

        result = graph.invoke({"product": product, "draft": "", "feedback": "", "attempts": 0})
        status = "passed validation" if not result["feedback"] else f"stopped at cap ({result['feedback']})"
        print(f"\n[{result['attempts']} attempt(s), {status}]")
        print(f"{result['draft']}\n")


if __name__ == "__main__":
    main()
