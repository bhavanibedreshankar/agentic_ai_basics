"""
CONCEPT: Working memory — a scratchpad the agent uses MID-TASK to track
intermediate state (partial results, running totals, what's been checked
so far), scoped to a single task and cleared once that task is done.

This is the odd one out among the memory types in this repo, and
deliberately so:
  - ../external_memory/, ../episodic_memory/, and ../semantic_memory/ all
    persist to disk and are meant to OUTLIVE the current task — that's
    the whole point of them.
  - Working memory does the opposite on purpose: `SCRATCHPAD` is cleared
    at the START of every new task (see clear_scratchpad in main()
    below), specifically so nothing leaks between unrelated tasks. It's
    also never written to a file — there's nothing here worth persisting
    once the task that produced it is finished.

It's also distinct from ../../Planning_and_Reasoning/plan_and_execute/,
which tracks intermediate step results too — but there, the HOST PROGRAM
manages that state (a plain Python `results` list the code appends to).
Here, the MODEL manages its own state, explicitly, via write_scratchpad
and read_scratchpad tool calls — the agent decides what's worth jotting
down and when to check back on it, rather than the surrounding code
deciding for it.

Demonstrated on a multi-step calculation (compound interest year by
year) where tracking the running balance in a structured scratchpad,
rather than trusting it stays consistent purely from conversational
context, means the intermediate numbers stay easy to inspect and
resistant to arithmetic drift across steps. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are an assistant that solves multi-step tasks using a "
    "scratchpad to track intermediate state. As you work through a "
    "problem, use write_scratchpad to record each intermediate result "
    "(e.g. a running total, a value you'll need again in a later step) "
    "under a clear key. Use read_scratchpad to check what you've already "
    "worked out before computing something again. Only give your final "
    "answer once the scratchpad reflects everything you've worked out."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: the scratchpad itself. A plain dict, module-level so
# write_scratchpad/read_scratchpad can share it — deliberately the
# simplest possible structure. Cleared per task in main(), never
# persisted to disk (see the module docstring for why).
SCRATCHPAD: dict[str, str] = {}


def write_scratchpad(key: str, value: str) -> str:
    SCRATCHPAD[key] = value
    return f"Scratchpad updated: {key} = {value}"


def read_scratchpad() -> str:
    if not SCRATCHPAD:
        return "Scratchpad is empty."
    return "\n".join(f"{key}: {value}" for key, value in SCRATCHPAD.items())


TOOLS = [
    {
        "name": "write_scratchpad",
        "description": "Record an intermediate value in the scratchpad under a key, for use in a later step of this task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "A short label for this value, e.g. 'balance_after_year_1'"},
                "value": {"type": "string", "description": "The value to record"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "read_scratchpad",
        "description": "Read back everything currently recorded in the scratchpad for this task.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "write_scratchpad":
        return write_scratchpad(**tool_input), False
    if name == "read_scratchpad":
        return read_scratchpad(), False
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

    print("Working memory demo. Type 'exit' to quit.\n")
    print(
        "Try: \"I invest $1000 at 5% annual compound interest. Track the "
        "balance at the end of each of the next 4 years in your "
        "scratchpad, then tell me the final balance.\"\n"
    )

    while True:
        task = input("Task: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue

        # CONCEPT: clear working memory at the start of every new task —
        # this is what makes it "working" memory rather than another
        # long-term store. Nothing from a previous task should leak into
        # the next one.
        SCRATCHPAD.clear()

        messages: list[dict] = [{"role": "user", "content": task}]
        run_turn(messages)

        print(f"  [final scratchpad state: {dict(SCRATCHPAD)}]\n")


if __name__ == "__main__":
    main()
