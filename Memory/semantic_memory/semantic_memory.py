"""
CONCEPT: Semantic memory — persistent knowledge about the world, domain
facts, or user preferences. Unlike ../episodic_memory/ (memory of specific
past EVENTS), semantic memory holds timeless, general statements: "the
user prefers metric units", "the user's role is backend engineer" — true
regardless of which conversation you're in or when it was learned.

This template's semantic memory is STRUCTURED and UPDATABLE, which is
what distinguishes it from ../memory_management/basic_agentic_memory.py
(a flat, append-only list of free-text strings, injected wholesale into
every system prompt). Real semantic knowledge is closer to a profile than
a list: facts belong to CATEGORIES ("preferences", "role", "constraints"),
are addressed by KEY, and — critically — get OVERWRITTEN when they
change, rather than piling up as a growing list of possibly-contradictory
statements ("user's favorite color is blue" ... "user's favorite color is
green" — semantic memory should end up believing green, not remembering
both forever).

Three tools give the model deliberate control over this structure:
remember (set or overwrite a key within a category), forget (remove a
key), and recall (read back a category, or everything). Type 'exit' to
end the conversation — memory persists in semantic_memory.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a helpful assistant with structured, persistent semantic "
    "memory of facts about the user. Use remember to store or update a "
    "fact (choosing a sensible category and key), forget to remove one "
    "that's no longer true, and recall to check what you already know "
    "before assuming. When a new fact contradicts an old one (e.g. a "
    "changed preference), overwrite it with remember rather than keeping "
    "both."
)

MEMORY_FILE = Path(__file__).parent / "semantic_memory.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def _load_memory() -> dict[str, dict[str, str]]:
    if not MEMORY_FILE.exists():
        return {}
    return json.loads(MEMORY_FILE.read_text())


def _save_memory(memory: dict[str, dict[str, str]]) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def remember(category: str, key: str, value: str) -> str:
    """CONCEPT: set-or-overwrite, not append. If (category, key) already
    exists, this REPLACES the old value rather than adding a second,
    possibly contradictory entry — semantic memory reflects the current
    state of the world, not a history of every belief ever held about it
    (that history is exactly what ../episodic_memory/ is for).
    """
    memory = _load_memory()
    memory.setdefault(category, {})
    previous = memory[category].get(key)
    memory[category][key] = value
    _save_memory(memory)
    if previous is not None and previous != value:
        return f"Updated {category}.{key}: '{previous}' -> '{value}'"
    return f"Remembered {category}.{key} = '{value}'"


def forget(category: str, key: str) -> str:
    memory = _load_memory()
    if category in memory and key in memory[category]:
        del memory[category][key]
        if not memory[category]:
            del memory[category]
        _save_memory(memory)
        return f"Forgot {category}.{key}"
    return f"No memory found at {category}.{key}"


def recall(category: str | None = None) -> str:
    """Read back what's stored — a specific category, or the whole
    profile if no category is given.
    """
    memory = _load_memory()
    if not memory:
        return "No memories stored yet."

    if category is not None:
        if category not in memory:
            return f"No memories stored under category '{category}'."
        return json.dumps({category: memory[category]}, indent=2)

    return json.dumps(memory, indent=2)


TOOLS = [
    {
        "name": "remember",
        "description": "Store or update a fact, organized by category and key. Overwrites any existing value at the same category/key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "A grouping for related facts, e.g. 'preferences', 'role', 'constraints'"},
                "key": {"type": "string", "description": "The specific fact name within the category, e.g. 'unit_system'"},
                "value": {"type": "string", "description": "The fact's value"},
            },
            "required": ["category", "key", "value"],
        },
    },
    {
        "name": "forget",
        "description": "Remove a specific fact by category and key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "The category the fact belongs to"},
                "key": {"type": "string", "description": "The key of the fact to remove"},
            },
            "required": ["category", "key"],
        },
    },
    {
        "name": "recall",
        "description": "Read back stored facts — a specific category, or everything if no category is given.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "Optional: limit recall to this category"}},
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name == "remember":
            return remember(**tool_input), False
        if name == "forget":
            return forget(**tool_input), False
        if name == "recall":
            return recall(**tool_input), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}", True


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
                print(f"  [result] {result_text}")
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

    existing = _load_memory()
    fact_count = sum(len(facts) for facts in existing.values())
    print(f"Semantic memory assistant — {fact_count} facts across {len(existing)} categories stored.")
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
