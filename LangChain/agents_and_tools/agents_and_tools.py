"""
CONCEPT: Agents and Tools — an LLM-driven loop that decides, on its own,
which tools to call and when to stop, rather than following a sequence
YOUR code laid out in advance (that's chaining — see ../chains/chains.py).

../../Core_Architecture/tool_use/basic_agentic_tools.py builds this loop by hand: a JSON-Schema
`TOOLS` list, a `client.messages.create(..., tools=TOOLS)` call, a check of
`response.stop_reason == "tool_use"`, manual dispatch to Python functions,
and a `while True` that keeps calling the API with tool results appended
until Claude stops asking for tools. LangChain's `create_agent` collapses
all of that into one call: give it a model and a list of `@tool`-decorated
functions, and it returns a ready-to-invoke agent (built internally on a
LangGraph state graph — see ../../LangGraph/ for that layer directly) that
runs the exact same request -> tool-call -> tool-result -> request loop
for you. The `@tool` decorator itself replaces the hand-written JSON Schema
dicts in basic_agentic_tools.py's `TOOLS` list — it derives the schema from
the function's type hints and docstring instead of you writing it out.

Use case: an expense reimbursement assistant with three tools (submit,
list, approve). Every message in `result["messages"]` after invoking is
printed, so the full tool-call trace is visible — proving the loop ran
even though this file never writes it explicitly. Type 'exit' to end the
conversation.
"""

from __future__ import annotations

import os
import sys
import uuid

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are an expense reimbursement assistant. Use the available tools "
    "to submit, list, and approve expenses. After using a tool, briefly "
    "confirm what happened."
)

# In-memory store, same role as the TASKS_FILE dict in
# ../../Core_Architecture/tool_use/basic_agentic_tools.py — a stand-in for a real database.
_EXPENSES: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# CONCEPT: @tool — derives a tool's name, JSON Schema, and description from
# the function signature (type hints -> schema types) and docstring
# (-> description) automatically. Compare this to the hand-written
# `input_schema` dicts in basic_agentic_tools.py's TOOLS list: same
# information, but generated from ordinary Python instead of duplicated by
# hand.
# ---------------------------------------------------------------------------
@tool
def submit_expense(description: str, amount: float) -> str:
    """Submit a new expense for reimbursement. Amount is in US dollars."""
    expense_id = uuid.uuid4().hex[:8]
    _EXPENSES[expense_id] = {"description": description, "amount": amount, "status": "pending"}
    return f"Submitted expense {expense_id}: {description} (${amount:.2f}), status=pending"


@tool
def list_expenses(status: str = "all") -> str:
    """List expenses, optionally filtered by status: 'pending', 'approved', or 'all'."""
    items = [
        f"{eid}: {e['description']} (${e['amount']:.2f}) [{e['status']}]"
        for eid, e in _EXPENSES.items()
        if status == "all" or e["status"] == status
    ]
    return "\n".join(items) if items else "No matching expenses."


@tool
def approve_expense(expense_id: str) -> str:
    """Approve a pending expense by its id."""
    if expense_id not in _EXPENSES:
        return f"Error: no expense with id '{expense_id}'"
    _EXPENSES[expense_id]["status"] = "approved"
    return f"Approved expense {expense_id}."


TOOLS = [submit_expense, list_expenses, approve_expense]


def build_agent(llm: BaseChatModel):
    # CONCEPT: this one call is the entire agent — no explicit loop, no
    # manual dispatch-by-name. `create_agent` wires the model, the tool
    # list, and the tool-calling loop together and returns a compiled,
    # invokable graph.
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def run_turn(agent, history: list) -> list:
    """Invoke the agent for one user turn and print every tool call and
    result it produced along the way, then return the updated message
    history (so the outer loop in main() can keep the conversation going
    across turns).
    """
    result = agent.invoke({"messages": history})
    new_messages = result["messages"]

    # CONCEPT: everything the agent did this turn — tool calls, tool
    # results, and the final reply — is right there in `new_messages`,
    # appended after whatever was already in `history`. Printing the slice
    # after the original history length makes the internal loop's work
    # visible, the same way the manual print statements in
    # basic_agentic_tools.py's run_turn() do, just reading it back after
    # the fact instead of printing as each step happens.
    for message in new_messages[len(history):]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  [tool] {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"  [result] {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            print(f"\nClaude: {message.content}\n")

    return new_messages


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    agent = build_agent(llm)

    print("Expense reimbursement agent. Type 'exit' to end the conversation.\n")

    history: list = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        history = run_turn(agent, history)


if __name__ == "__main__":
    main()
