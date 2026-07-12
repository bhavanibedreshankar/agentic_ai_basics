"""
CONCEPT: Context pruning — selectively removing content that's no longer
useful from the conversation history, to keep the context window focused
and cut token cost, WITHOUT losing the conversation's actual flow.

This is a more surgical technique than ../../Memory/memory_management/basic_agentic_memory.py's
trim_history, which drops whole old TURNS wholesale (a blunt sliding
window). Pruning instead targets specific, bulky pieces of content that
have already served their purpose — most commonly, tool results. Once
Claude has read a tool's output and acted on it, the full raw output
rarely needs to stick around; a short placeholder is enough to keep the
conversation coherent while freeing up the tokens it was using.

Use case: a research assistant that repeatedly calls a (mocked)
search_web tool. Search results are large. Once a tool result has been
superseded by a NEWER one, its raw content is pruned down to a short
placeholder — the surrounding conversation (questions, Claude's answers)
is left completely untouched.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a research assistant. Use the search_web tool to look things "
    "up rather than answering from memory."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def search_web(query: str) -> str:
    """A stand-in for a real search tool. Returns a sizeable chunk of fake
    'search results' text, so there's something meaningful for pruning to
    trim in this demo — a real search tool's output would be just as
    bulky, if not bulkier.
    """
    return (
        f"Search results for '{query}':\n"
        f"1. Overview of {query} — a detailed explanation covering history, "
        f"key concepts, and common misconceptions about {query}.\n"
        f"2. {query} in practice — real-world examples and case studies "
        f"showing how {query} is used, with pros and cons discussed at length.\n"
        f"3. Frequently asked questions about {query}, including edge cases "
        f"and advanced considerations that most people don't need day to day.\n"
    )


TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information on a topic. Returns raw search result text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "search_web":
        return search_web(**tool_input), False
    return f"Unknown tool: {name}", True


# ---------------------------------------------------------------------------
# CONCEPT: selective pruning
# ---------------------------------------------------------------------------
PRUNE_PLACEHOLDER = "[pruned: earlier tool result no longer shown to save context — {chars} characters removed]"


def _tool_result_chars(messages: list[dict]) -> int:
    """Sum the character count of every tool_result block currently in the
    history — used only to show the before/after effect of pruning below.
    """
    total = 0
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                total += len(block.get("content") or "")
    return total


def prune_old_tool_results(messages: list[dict], keep_last_n: int = 1) -> None:
    """Replace every tool_result block except the most recent `keep_last_n`
    with a short placeholder, in place.

    CONCEPT: this only touches tool_result blocks (the raw, bulky output a
    tool produced) — it never touches the user's questions or Claude's own
    text responses, which is what makes pruning "surgical" rather than a
    blunt cut across whole turns. Claude can still see that a search
    happened and roughly what it returned (via its own earlier summary of
    the result, if it gave one), just not the full raw text anymore.
    """
    tool_result_blocks = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_result_blocks.append(block)

    stale_blocks = tool_result_blocks[:-keep_last_n] if keep_last_n > 0 else tool_result_blocks
    for block in stale_blocks:
        original = block["content"]
        if isinstance(original, str) and not original.startswith("[pruned"):
            block["content"] = PRUNE_PLACEHOLDER.format(chars=len(original))


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
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

        # CONCEPT: prune right after a new tool result lands — any OLDER
        # tool results are now superseded and safe to prune.
        before = _tool_result_chars(messages)
        prune_old_tool_results(messages, keep_last_n=1)
        after = _tool_result_chars(messages)
        if before != after:
            print(f"  [pruned tool results: {before} -> {after} characters]")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Research assistant (context pruning demo). Type 'exit' to end the conversation.\n")

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
