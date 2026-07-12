"""
CONCEPT: Max Iterations / Turn Limit — a hard, unconditional safety cap
on how many loop cycles an agent may run before stopping automatically,
independent of whether the task looks finished, stuck, or fine.

This is the SIMPLEST and MOST BLUNT safety mechanism in `Execution_Loops/`
— worth understanding precisely because of how it differs from the two
more targeted mechanisms next to it:
  - `../human_in_the_loop/human_in_the_loop.py` pauses for approval on
    SPECIFIC tools, every single time, regardless of how long the task
    has been running.
  - `../interrupts_breakpoints/interrupts_breakpoints.py` pauses only
    when a specific CONDITION is met (cost, a flagged action, a detected
    stall) — it can, in principle, let a task run forever if none of its
    conditions ever trigger.
  - A max-iteration cap has NO condition to evaluate at all. It counts
    loop iterations and stops at a fixed number, full stop — the
    correct backstop for exactly the failure mode the other two can
    miss: a bug, a bad prompt, or an unanticipated situation that causes
    a loop with no matching breakpoint to spin indefinitely, burning
    tokens the whole time.

This template runs the SAME agentic tool-calling loop shape used
throughout the repo, with one added piece of bookkeeping: an iteration
counter, checked at the top of every cycle, that forces the loop to stop
and report itself once `MAX_ITERATIONS` is reached — even mid-task, even
if Claude still wants to keep calling tools. To make the cap something
you can actually observe hitting (rather than trusting it exists), this
template's mock tool is deliberately unhelpful in a way that tends to
make Claude retry a few times before giving up.

Use case: a "flaky" search tool that sometimes returns unhelpful results,
demonstrated with a MAX_ITERATIONS low enough to hit in a normal session.
Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a research assistant with a search tool. The tool is "
    "sometimes unhelpful — if a search doesn't return what you need, try "
    "rephrasing the query and searching again."
)

# CONCEPT: the cap itself. Deliberately small here so it's reachable in a
# normal demo session — a real agent would set this much higher (tens or
# hundreds of iterations), sized to the task, not to make a demo work.
MAX_ITERATIONS = 5

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# A tool that deliberately gives an unhelpful, generic answer to ANY
# query that isn't an exact match — standing in for any real-world tool
# that sometimes doesn't return what's needed, which is what makes a
# runaway retry loop a real (not just theoretical) risk worth capping.
KNOWN_QUERIES = {"claude api pricing"}


def flaky_search(query: str) -> str:
    if query.strip().lower() in KNOWN_QUERIES:
        return "Found it: pricing details are published at the official pricing page."
    return "No useful results found for that query. Try rephrasing."


TOOLS = [
    {
        "name": "flaky_search",
        "description": "Search for information. May not return useful results on the first try.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "flaky_search":
        return flaky_search(**tool_input), False
    return f"Unknown tool: {name}", True


def run_turn(messages: list[dict], max_iterations: int = MAX_ITERATIONS) -> str:
    """CONCEPT: the cap enforced. `iteration` increments once per loop
    pass; the check happens at the TOP of the loop, before making the API
    call for that pass, so the Nth call never happens once the limit is
    reached — the loop returns a clear status instead of silently
    stopping or, worse, running one call over budget.
    """
    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iterations:
            return f"stopped: reached MAX_ITERATIONS ({max_iterations}) without a final answer"

        print(f"  [iteration {iteration}/{max_iterations}]")
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
            return "completed: Claude gave a final answer"

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

    print(f"Max iterations demo (cap: {MAX_ITERATIONS} loop cycles). Type 'exit' to quit.\n")
    print("Try: \"Look up the shipping policy for international orders.\" (not a known query — likely to hit the cap)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages: list[dict] = [{"role": "user", "content": user_input}]
        status = run_turn(messages)
        print(f"\n=== {status} ===\n")


if __name__ == "__main__":
    main()
