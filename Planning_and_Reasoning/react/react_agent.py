"""
CONCEPT: ReAct (Reason + Act) — a prompting pattern where the model
explicitly alternates between verbalizing its REASONING and taking an
ACTION (a tool call), observing the result, and reasoning again — a
Thought -> Action -> Observation cycle, repeated until it has enough
information to give a Final Answer.

This is structurally similar to ../../Core_Architecture/tool_use/basic_agentic_tools.py's
agentic loop — both call tools and feed results back. The difference is
what ReAct adds: it makes the reasoning BETWEEN actions explicit and
visible in the transcript, not just an implicit step the model does
silently before deciding to call a tool. That matters for two reasons:
  1. Auditability — you can see WHY the model chose an action, not just
     which action it chose. If it goes wrong, the labeled Thought tells
     you where the reasoning broke down.
  2. Better multi-step performance — the original ReAct paper (Yao et
     al., 2022) found that forcing the model to articulate a plan before
     each action measurably improved multi-step task success compared to
     acting without stated reasoning.

This template gets ReAct's Thought/Action/Observation structure via
PROMPTING — instructing the model, in the system prompt, to narrate a
Thought before every tool call — which is what "ReAct" originally means:
a prompting pattern, not an API feature. (Claude's native extended
thinking, shown in ../chain_of_thought/chain_of_thought.py, is a related
but distinct mechanism — it also surfaces reasoning, but as separate
`thinking` blocks rather than as part of the visible Thought/Action
narrative this pattern relies on.)

Use case: a research agent with two tools — lookup_fact and calculate —
answering questions that need multiple lookups and a calculation to
solve, so the Thought -> Action -> Observation cycle repeats more than
once per question. Type 'exit' to end the conversation.
"""

from __future__ import annotations

import ast
import operator
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"

# CONCEPT: this is the ReAct pattern's core instruction. It doesn't just
# tell Claude which tools exist — it tells Claude HOW to reason about
# using them: narrate a Thought, take one Action, wait for the
# Observation, and repeat, only answering once it actually has what it
# needs.
SYSTEM_PROMPT = (
    "You are a research assistant that solves problems using explicit "
    "step-by-step reasoning interleaved with tool use, following this "
    "pattern:\n"
    "  Thought: reason about what you know and what you still need to find out.\n"
    "  Action: call exactly one tool to make progress.\n"
    "  (you will then receive an Observation with the tool's result)\n"
    "Repeat Thought/Action/Observation as many times as needed. Always "
    "write a Thought before every tool call, explaining why you're "
    "calling it. Once you have everything you need, respond with a final "
    "answer and no further tool calls."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# Tools: a small fact database (forcing multiple lookups for most
# interesting questions) plus a calculator (forcing a reasoning step to
# combine looked-up facts).
# ---------------------------------------------------------------------------
FACTS = {
    "population of france": "France has a population of approximately 68 million.",
    "population of germany": "Germany has a population of approximately 84 million.",
    "population of japan": "Japan has a population of approximately 124 million.",
    "area of france": "France covers approximately 551,700 square kilometers.",
    "area of germany": "Germany covers approximately 357,600 square kilometers.",
    "price of a widget": "A widget costs $12.50.",
    "price of a gadget": "A gadget costs $34.00.",
}


def lookup_fact(topic: str) -> str:
    key = topic.strip().lower()
    if key in FACTS:
        return FACTS[key]
    # Fuzzy fallback: word-overlap match, so a differently-worded lookup
    # (e.g. "france population" instead of "population of france") still
    # finds the right fact rather than dead-ending the whole chain.
    key_words = set(key.split())
    best_topic, best_overlap = None, 0
    for known_topic in FACTS:
        overlap = len(key_words & set(known_topic.split()))
        if overlap > best_overlap:
            best_topic, best_overlap = known_topic, overlap
    if best_topic:
        return FACTS[best_topic]
    return f"No fact found for '{topic}'. Known topics: {', '.join(FACTS)}"


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval").body
    return f"{expression} = {_safe_eval(tree)}"


TOOLS = [
    {
        "name": "lookup_fact",
        "description": "Look up a known fact by topic, e.g. 'population of france' or 'price of a widget'.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "The topic to look up"}},
            "required": ["topic"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression, e.g. '68000000 / 84000000'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "The arithmetic expression to evaluate"}},
            "required": ["expression"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name == "lookup_fact":
            return lookup_fact(**tool_input), False
        if name == "calculate":
            return calculate(**tool_input), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}", True


def run_turn(messages: list[dict]) -> None:
    """The ReAct loop. Structurally the same shape as
    ../../Core_Architecture/tool_use/basic_agentic_tools.py's run_turn — call the API,
    check for tool calls, execute them, feed results back, repeat. What
    makes THIS a ReAct loop rather than plain tool use is entirely in the
    SYSTEM_PROMPT above: it's what makes Claude narrate a labeled Thought
    before each Action instead of just silently deciding to call a tool.
    """
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
            if block.type == "text" and block.text.strip():
                # Claude's own text should already contain the "Thought:"
                # label per the system prompt instructions — print as-is.
                print(f"\n{block.text.strip()}")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"Action: {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"Observation: {result_text}")
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

    print("ReAct research agent. Type 'exit' to end the conversation.\n")
    print("Try: \"What's the combined population of France and Germany, and how much would 3 widgets and 2 gadgets cost?\"\n")

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
        print()


if __name__ == "__main__":
    main()
