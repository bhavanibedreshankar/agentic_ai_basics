"""
CONCEPT: Skill / Tool Registry — a catalog of available tools, with
descriptions, that the LLM uses to decide which one (if any) to call.

../../Core_Architecture/tool_use/basic_agentic_tools.py hand-maintains tool metadata in TWO separate places:
a `TOOLS` list (what Claude sees) and an `execute_tool()` if/elif chain
(what actually runs). That works fine for 3 tools, but it doesn't scale —
add a 20th tool and it's easy to update one place and forget the other,
and the if/elif chain gets long and repetitive.

A REGISTRY fixes this by making each tool's name, description, JSON
schema, AND handler function all part of ONE record, stored in a single
catalog (a dict, here). Everything downstream is generated FROM that
catalog:
  - The `tools` list sent to the API is built by reading every entry's
    description and schema.
  - Dispatching a tool call is a dict lookup by name, not an if/elif chain.

This means adding a new tool is a single, self-contained step (register
it) with no other code to touch — and there's no way for the catalog and
the dispatch logic to drift out of sync, because they're the same data.

Use case: a small utility toolbox agent (calculator, word counter, text
tools, clock) — enough distinct tools to make the registry pattern's value
obvious. Type 'exit' to end the conversation.
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic

# --- API settings (see ../../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant with access to a small set of utility tools. Use them when they'd give a more accurate answer than reasoning alone."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: The registry itself — a catalog mapping tool name -> its full
# record. This dict IS the "skill/tool registry" the topic refers to.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, input_schema: dict):
    """Decorator that adds a function to TOOL_REGISTRY as a callable tool.

    CONCEPT: single source of truth. The tool's name, its description (what
    the LLM reads to decide WHEN to use it), its JSON Schema (what the LLM
    reads to decide HOW to call it), and the actual handler function that
    runs when it's called are all declared together, right here, in one
    place — not split across a separate list and a separate dispatch
    function like in ../../Core_Architecture/tool_use/basic_agentic_tools.py.
    """

    def decorator(func):
        TOOL_REGISTRY[name] = {
            "description": description,
            "input_schema": input_schema,
            "handler": func,
        }
        return func

    return decorator


def build_tool_catalog() -> list[dict]:
    """CONCEPT: the registry drives what Claude sees. This is the `tools`
    list passed to the API — generated directly from TOOL_REGISTRY instead
    of hand-maintained separately. Register a new tool below and it
    automatically appears here; nothing else needs to change.
    """
    return [
        {"name": name, "description": entry["description"], "input_schema": entry["input_schema"]}
        for name, entry in TOOL_REGISTRY.items()
    ]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """CONCEPT: registry-driven dispatch. Look the handler up by name — a
    single dict lookup — instead of an if/elif chain that grows with every
    tool. This stays exactly as simple whether the registry holds 5 tools
    or 500.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Unknown tool: {name}", True
    try:
        result = entry["handler"](**tool_input)
        return str(result), False
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to Claude
        return f"Error: {exc}", True


# ---------------------------------------------------------------------------
# The tools themselves. Each one is registered where it's defined — the
# metadata Claude sees lives right next to the code that actually runs.
# ---------------------------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    """Walk a parsed expression tree, allowing only basic arithmetic.
    Never pass model-influenced strings straight to eval() — even though
    this input comes from Claude rather than directly from the user, it
    could still be manipulated (e.g. via a crafted prompt), so we restrict
    what's evaluable rather than trusting the string outright.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@register_tool(
    name="calculate",
    description="Evaluate a basic arithmetic expression. Call this for any math question instead of computing it yourself.",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 + 2 * 3'"},
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval").body
    return f"{expression} = {_safe_eval(tree)}"


@register_tool(
    name="word_count",
    description="Count the words and characters in a piece of text. Call this when the user asks how long a text is.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to measure"}},
        "required": ["text"],
    },
)
def word_count(text: str) -> str:
    return f"{len(text.split())} words, {len(text)} characters"


@register_tool(
    name="reverse_text",
    description="Reverse a piece of text character by character.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to reverse"}},
        "required": ["text"],
    },
)
def reverse_text(text: str) -> str:
    return text[::-1]


@register_tool(
    name="convert_case",
    description="Convert text to uppercase, lowercase, or title case.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to convert"},
            "case": {"type": "string", "enum": ["upper", "lower", "title"], "description": "The case to convert to"},
        },
        "required": ["text", "case"],
    },
)
def convert_case(text: str, case: str) -> str:
    return {"upper": text.upper, "lower": text.lower, "title": text.title}[case]()


@register_tool(
    name="get_current_time",
    description="Get the current date and time, optionally in a specific IANA timezone (e.g. 'America/New_York'). Defaults to UTC.",
    input_schema={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'America/New_York'. Defaults to UTC."},
        },
    },
)
def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return f"Unknown timezone '{timezone}'. Try an IANA name like 'America/New_York'."
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# NOTE on scale: with just 5 tools, sending the whole catalog on every
# request (build_tool_catalog()) is cheap. A real system with dozens or
# hundreds of tools would instead want to send only the tools relevant to
# the current request — Anthropic's "tool search" feature does exactly
# that: the registry stays large, but Claude discovers and loads only the
# tool definitions it actually needs for a given task, keeping the request
# small. The registry pattern here is what makes that upgrade possible
# later without restructuring anything — the catalog already exists as
# one addressable collection.


# ---------------------------------------------------------------------------
# The agentic loop — identical shape to ../../Core_Architecture/tool_use/basic_agentic_tools.py, but tool
# definitions and dispatch both come from the registry instead of being
# hardcoded here.
# ---------------------------------------------------------------------------

def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=build_tool_catalog(),  # <-- generated from the registry
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
                result_text, is_error = execute_tool(block.name, block.input)  # <-- registry dispatch
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

    print(f"Utility agent — {len(TOOL_REGISTRY)} tools registered: {', '.join(TOOL_REGISTRY)}")
    print("Type 'exit' to end the conversation.\n")

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
