"""
CONCEPT: State Machine — a formal structure defining an agent's VALID
states and the ALLOWED transitions between them, so an invalid action
(shipping an order that hasn't been paid for, refunding an order that
was never placed) is rejected by CODE before it ever reaches the model,
rather than relying on the model to remember the rules from a prompt.

None of the multi-step templates elsewhere in this repo enforce this. `../../Multi_Agent_Systems/agent_handoff/agent_handoff.py`
tracks a `current_agent_name` string too, but ANY transfer tool can fire
from ANY agent — nothing stops triage from "transferring" to itself or a
specialist from transferring to another specialist, because there's no
table of which transitions are even legal. `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`
executes steps in a fixed order decided once at planning time — there's
no notion of a current state a step could be validated against, and no
way to reject an attempt to run step 3 before step 1.

This template makes the rules explicit and enforces them in TWO places:
  1. The `TRANSITIONS` table — a plain dict of `{state: {allowed next
     states}}` — is the single source of truth for what's legal.
  2. `execute_tool` checks every requested transition against that table
     BEFORE calling the state-changing function, and returns a tool
     ERROR (not a crash, not a silent no-op) if the model requests an
     illegal one — the same "tool errors are recoverable, not fatal"
     pattern used throughout this repo, just enforcing state-legality
     instead of a malformed argument.

Use case: an order-processing agent whose order object moves through
pending -> paid -> shipped -> delivered (with a cancelled branch off
pending or paid). Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are an order processing assistant. Use the order management "
    "tools to move an order through its lifecycle. Check get_order_state "
    "before attempting a transition if you're not sure what's currently "
    "allowed — not every action is valid from every state."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the state machine definition itself. A state machine is
# fully specified by (1) its set of states and (2) which transitions
# between them are legal — everything else in this file is just
# enforcing and exposing that table.
# ---------------------------------------------------------------------------
TRANSITIONS: dict[str, set[str]] = {
    "pending": {"paid", "cancelled"},
    "paid": {"shipped", "cancelled"},
    "shipped": {"delivered"},
    "delivered": set(),   # terminal state — no legal transitions out
    "cancelled": set(),   # terminal state — no legal transitions out
}


class Order:
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        self.state = "pending"
        self.history: list[str] = ["pending"]

    def can_transition_to(self, new_state: str) -> bool:
        return new_state in TRANSITIONS.get(self.state, set())

    def transition_to(self, new_state: str) -> None:
        """CONCEPT: mutation only ever happens through this one method,
        and this method never checks legality itself — by the time
        anything calls transition_to, execute_tool has already verified
        it via can_transition_to. Two enforcement points would be
        redundant; ONE clear place (the tool dispatch layer, checked
        BEFORE the state-changing call) is enough as long as nothing
        bypasses it.
        """
        self.state = new_state
        self.history.append(new_state)


ORDERS: dict[str, Order] = {}


def get_order_state(order_id: str) -> str:
    order = ORDERS.setdefault(order_id, Order(order_id))
    allowed = sorted(TRANSITIONS[order.state])
    return f"Order {order_id} is currently '{order.state}'. Allowed next states: {allowed or 'none (terminal state)'}."


def transition_order(order_id: str, new_state: str) -> str:
    order = ORDERS.setdefault(order_id, Order(order_id))

    if new_state not in TRANSITIONS:
        return f"Error: '{new_state}' is not a valid state. Valid states: {sorted(TRANSITIONS)}."

    # CONCEPT: this is the enforcement point. The check happens BEFORE
    # order.transition_to() is ever called — an illegal request never
    # touches the actual state at all, not even briefly.
    if not order.can_transition_to(new_state):
        return (
            f"Error: cannot transition order {order_id} from '{order.state}' to "
            f"'{new_state}'. Allowed from '{order.state}': {sorted(TRANSITIONS[order.state]) or 'none (terminal state)'}."
        )

    order.transition_to(new_state)
    return f"Order {order_id} transitioned to '{new_state}'. History: {' -> '.join(order.history)}."


TOOLS = [
    {
        "name": "get_order_state",
        "description": "Check an order's current state and which transitions are legal from there.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "transition_order",
        "description": "Attempt to move an order to a new state. Will be rejected if the transition isn't legal from the order's current state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID"},
                "new_state": {
                    "type": "string",
                    "enum": sorted(TRANSITIONS),
                    "description": "The state to move the order to",
                },
            },
            "required": ["order_id", "new_state"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "get_order_state":
        return get_order_state(**tool_input), False
    if name == "transition_order":
        result = transition_order(**tool_input)
        return result, result.startswith("Error:")
    return f"Unknown tool: {name}", True


def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [result{'  (REJECTED)' if is_error else ''}] {result_text}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Order processing state machine demo. Type 'exit' to quit.\n")
    print("Try: \"Ship order A123.\" (should be rejected — it hasn't been paid for yet)\n")
    print("Try: \"Mark order A123 as paid, then ship it.\"\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
