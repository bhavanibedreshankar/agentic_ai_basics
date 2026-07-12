"""
CONCEPT: Memory management in AI agents — giving an agent both short-term
memory (the current conversation) and long-term memory (facts that persist
across separate runs of the program), and keeping both from growing
unbounded.

This template is a personal assistant that remembers facts about you across
sessions. It demonstrates three distinct memory techniques:

  1. SHORT-TERM MEMORY — the `messages` list, same idea as
     ../../Execution_Loops/agentic_loop/basic_agentic_loop.py. Lives only for this run of the program; gone the
     moment the process exits.

  2. LONG-TERM MEMORY — facts persisted to memory.json. Survives between
     runs: close the program, reopen it, and Claude still "remembers" you,
     because we load saved facts and inject them into the system prompt.

  3. MEMORY OPTIMIZATION — short-term memory grows every turn, which costs
     more tokens (and risks the context window limit) the longer a chat
     runs. This template bounds it with a sliding window (trim_history) and
     shows how to actually measure context size with the token-counting
     endpoint, rather than guessing.

Type 'exit' to end the conversation. Long-term memories persist after exit;
short-term conversation history does not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: Long-term memory — persisted storage
# ---------------------------------------------------------------------------
# A plain JSON file is the simplest possible long-term memory store: just a
# list of short facts. Real agents might use a database or a vector store
# instead, but the pattern is identical — write facts out, read them back in
# on the next run, and inject them into context so the model "knows" them
# again. Claude itself never remembers anything between API calls; all
# memory is something YOUR code manages and feeds back in.
MEMORY_FILE = Path(__file__).parent / "memory.json"


def _load_memories() -> list[str]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def _save_memories(memories: list[str]) -> None:
    MEMORY_FILE.write_text(json.dumps(memories, indent=2))


def build_system_prompt(memories: list[str]) -> str:
    """CONCEPT: context injection — this is HOW long-term memory actually
    reaches the model. Claude has no memory of past runs on its own, so we
    read whatever's saved and inject it as plain text into the system
    prompt at the top of every request. From Claude's perspective, this
    looks identical to information the user just told it.

    Because this is called fresh on every API call (see run_turn below),
    a memory saved earlier in THIS SAME conversation is immediately folded
    back in too — not just in future runs of the program.
    """
    base = (
        "You are a helpful personal assistant with long-term memory. "
        "When the user shares something worth remembering for future "
        "conversations (their name, preferences, ongoing projects, etc.), "
        "call the save_memory tool to store it. Don't bother saving trivial "
        "or one-off details."
    )
    if not memories:
        return base
    recalled = "\n".join(f"- {m}" for m in memories)
    return f"{base}\n\nWhat you remember about this user from past conversations:\n{recalled}"


# ---------------------------------------------------------------------------
# CONCEPT: Short-term memory optimization — bounding context growth
# ---------------------------------------------------------------------------
# MAX_HISTORY_TURNS caps how many messages we keep in the in-session
# `messages` list before trimming. Without a cap, a long conversation keeps
# resending its ENTIRE history on every single API call — cost and latency
# both grow with every turn, and eventually you'd hit the model's context
# window limit. A sliding window (keep only the N most recent messages) is
# the simplest fix.
#
# It's also lossy: anything older than the window is gone, and Claude can no
# longer refer back to it. A more sophisticated version of this same idea
# would SUMMARIZE the dropped turns (via a smaller, cheaper API call) and
# keep the summary instead of discarding history outright — compressing
# context rather than losing it. Anthropic's built-in "compaction" feature
# is a production-grade version of exactly that pattern.
MAX_HISTORY_TURNS = 12  # keep the most recent 12 messages (~6 user/assistant pairs)


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep only the most recent MAX_HISTORY_TURNS messages."""
    if len(messages) <= MAX_HISTORY_TURNS:
        return messages
    return messages[-MAX_HISTORY_TURNS:]


def print_context_size(messages: list[dict]) -> None:
    """CONCEPT: measuring memory usage. Token count — not the number of
    messages or characters — is what actually determines API cost and how
    close a conversation is to the model's context window limit. Use the
    count_tokens endpoint to get the real number instead of guessing from
    len(messages) or character counts, which can be very inaccurate.
    """
    count = client.messages.count_tokens(model=MODEL, messages=messages)
    print(f"  [context: ~{count.input_tokens} tokens across {len(messages)} messages]")


# ---------------------------------------------------------------------------
# The one tool this agent has: an explicit way for Claude to write to
# long-term memory. (See ../../Tools_and_Actions/tool_use/basic_agentic_tools.py for a deeper walkthrough of
# how tool definitions and the tool-calling loop work in general.)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "Save a short fact about the user to long-term memory, so it can "
            "be recalled in future conversations. Use this for durable facts "
            "(name, preferences, goals) — not for one-off details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember, written as a short standalone sentence",
                },
            },
            "required": ["fact"],
        },
    },
]


def execute_tool(name: str, tool_input: dict, memories: list[str]) -> tuple[str, bool]:
    if name == "save_memory":
        fact = tool_input["fact"]
        # Mutating `memories` in place means build_system_prompt sees the
        # update immediately on the very next API call, later in this
        # same run_turn loop.
        memories.append(fact)
        _save_memories(memories)
        return f"Saved to long-term memory: {fact}", False
    return f"Unknown tool: {name}", True


def run_turn(messages: list[dict], memories: list[str]) -> None:
    """Handle one user turn end-to-end, including any save_memory tool
    calls. Mutates `messages` in place, same pattern as
    ../../Tools_and_Actions/tool_use/basic_agentic_tools.py's run_turn.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_prompt(memories),
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
                result_text, is_error = execute_tool(block.name, block.input, memories)
                print(f"  [memory] {result_text}")
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

    # CONCEPT: recalling long-term memory at startup — this is what makes
    # memory feel persistent across separate runs of the program, even
    # though each run starts with a completely empty `messages` list.
    memories = _load_memories()
    if memories:
        print(f"(Recalling {len(memories)} saved memories from past sessions)\n")
    print("Chat with your assistant. Type 'exit' to end the conversation.\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! Long-term memories are saved — I'll remember them next time.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # CONCEPT: trim BEFORE the tool-calling loop for this turn starts,
        # not during it. A tool_use message and its matching tool_result
        # must always travel together — trimming mid-tool-loop risks
        # separating them, which the API will reject on the next call.
        messages = trim_history(messages)

        run_turn(messages, memories)
        print_context_size(messages)


if __name__ == "__main__":
    main()
