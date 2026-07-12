"""
CONCEPT: Agent — an AI system that autonomously perceives, reasons, plans,
and acts to achieve a GOAL, beyond simple question-answering.

Contrast with ../basics/basic.py's ask_claude: you send one message, you get
one response back, and that's the whole interaction — there's no notion of
"working on it" over multiple steps. This file's pursue_goal() is different
in kind, not just in size: you hand it ONE goal, and it decides, entirely on
its own, how many steps the task needs, which tool(s) to call and in what
order, when it has enough information, and when the goal is actually done —
all without a human approving or prompting each individual step.

That's also the difference from ../tool_use/basic_agentic_tools.py, which
DOES use tools in a loop, but waits for a fresh human message before every
single action (it's reactive, turn by turn). Here, one call to pursue_goal()
runs the entire perceive -> reason -> plan -> act cycle, repeatedly, until
the goal is met or MAX_STEPS is hit — the "loop" is the agent working
unattended, not a chat.

Use case: a budget-checking agent. Given a goal like "do these purchases fit
under my budget?", it has to plan a sequence of calculations (no single tool
call answers the goal directly), track intermediate findings itself, and
decide for itself when it has enough to give a final answer.
"""

from __future__ import annotations

import ast
import operator
import os
import sys

import anthropic

# --- API settings (see ../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are an autonomous agent working toward a goal, not a chat assistant "
    "answering one message. You will not get any further input from a human "
    "until you finish. Break the goal into steps yourself, use the 'calculate' "
    "and 'record_finding' tools as needed to work through them, and only stop "
    "calling tools once you're ready to give a final, complete answer to the "
    "original goal."
)

# MAX_STEPS: an unconditional cap on how many perceive-reason-act cycles the
# agent may run, same idea as ../../Execution_Loops/max_iterations/max_iterations.py.
# Autonomy without a backstop like this is dangerous — a confused agent
# could otherwise loop on tool calls indefinitely, burning tokens and money.
MAX_STEPS = 8

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# Tools available to the agent while it pursues a goal.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression. Call this for any math instead of computing it yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression, e.g. '12.50 + 7.25 + 19.99'"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "record_finding",
        "description": (
            "Record an intermediate finding or conclusion you've reached so far, before continuing. "
            "Call this after each meaningful step of your plan, not just at the end — it's how your "
            "reasoning gets made visible as you work, not only in your final answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "A short statement of what you've just figured out"},
            },
            "required": ["finding"],
        },
    },
]

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    # Never eval() a model-influenced string directly — restrict to basic
    # arithmetic on a parsed expression tree instead. Same approach as
    # ../../Agent_Frameworks_and_Patterns/tool_registry/basic_tool_registry.py's calculate tool.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Dispatch a tool call by name and return (result_text, is_error)."""
    try:
        if name == "calculate":
            tree = ast.parse(tool_input["expression"], mode="eval").body
            return f"{tool_input['expression']} = {_safe_eval(tree)}", False
        if name == "record_finding":
            # This tool has no real computation to do — its only purpose is
            # to make the agent's planning visible in the transcript. The
            # "result" just acknowledges the finding was noted.
            return f"Noted: {tool_input['finding']}", False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the agent
        return f"Error: {exc}", True


# ---------------------------------------------------------------------------
# CONCEPT: the autonomous perceive -> reason -> plan -> act cycle
# ---------------------------------------------------------------------------

def pursue_goal(goal: str) -> str:
    """Run the agent to completion on a single goal, with no further human
    input, and return its final answer.

    Each iteration of this loop is one full cycle:
      - ACT:     the agent's previous tool call (or nothing, on step 1)
      - PERCEIVE: it observes the tool's result
      - REASON:   the model decides what that result means
      - PLAN:     and what to do next — another tool call, or a final answer

    Unlike ../tool_use/basic_agentic_tools.py's run_turn(), there is no
    human supplying the next message between cycles — messages only grow
    from the agent's own tool calls and their results.
    """
    messages: list[dict] = [{"role": "user", "content": goal}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # response.stop_reason != "tool_use" means the agent has decided,
        # on its own, that it's done — this is the "knows when to stop"
        # half of autonomy. Nothing in this script tells it the goal is met.
        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            print(f"[step {step}] agent finished on its own\n")
            return final_text

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[step {step}] reasoning: {block.text.strip()}")
            if block.type == "tool_use":
                print(f"[step {step}] act: {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"[step {step}] perceive: {result_text}\n")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    # The unconditional backstop: if MAX_STEPS is hit, stop regardless of
    # whether the agent thinks it's done. Better an honest "ran out of
    # steps" than a silent infinite loop.
    return f"(stopped after {MAX_STEPS} steps without reaching a final answer)"


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Autonomous goal-pursuing agent. Give it one goal; it runs unattended until done.\n")
    goal = input(
        "Goal: "
    ).strip() or (
        "I want to buy 3 items priced at $12.50, $7.25, and $19.99. "
        "My budget is $50. Do they fit, and how much would I have left over?"
    )

    print(f"\nGoal: {goal}\n")
    answer = pursue_goal(goal)
    print(f"Final answer: {answer}")


if __name__ == "__main__":
    main()
