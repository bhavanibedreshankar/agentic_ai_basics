"""
CONCEPT: Interrupts / Breakpoints — predefined CONDITIONS, checked every
loop iteration, that pause execution and surface control to a human or
supervisor when triggered — distinct from HITL's per-action approval
gate in an important way: a breakpoint doesn't ask permission before
EVERY tool call, only when something SPECIFIC about the current state
crosses a threshold you defined in advance.

Contrast with `../human_in_the_loop/human_in_the_loop.py`: there, the
gate is keyed on the TOOL NAME — every `send_email` call, no matter what
it contains, pauses for approval. Here, the gate is keyed on CONDITIONS
over the running state — cost exceeding a budget, a specific tool
appearing at all (e.g. anything destructive), or a loop that's spinning
without making progress. Some breakpoints here would never fire even on
a `send_email`-shaped tool if the condition doesn't match; HITL's gate
fires on that tool 100% of the time regardless of content.

Three concrete breakpoint conditions are implemented, each checked once
per loop iteration in `check_breakpoints`, any one of which halts and
asks a human to continue or abort:
  1. COST breakpoint — cumulative token spend crosses a budget.
  2. ACTION breakpoint — a specific tool call is requested at all (here,
     `delete_file`, standing in for "anything irreversible").
  3. PROGRESS breakpoint — the same tool gets called with the same
     arguments repeatedly, suggesting the agent is stuck in a loop
     rather than converging on an answer.

Use case: a research agent (with a mock delete_file tool included
specifically to demonstrate the action breakpoint) working through a
multi-step task. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a research and file-management assistant. Use lookup_fact "
    "to research topics and delete_file to remove files when asked. Work "
    "through multi-step requests one tool call at a time."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: breakpoint thresholds — the whole policy lives here, as plain
# data, not scattered through conditional logic.
# ---------------------------------------------------------------------------
COST_BUDGET_TOKENS = 2000          # cumulative input+output tokens before pausing
ACTION_BREAKPOINTS = {"delete_file"}  # tool names that always pause, regardless of args
PROGRESS_STALL_THRESHOLD = 3        # identical (tool, args) calls in a row before pausing

FACTS = {
    "population of france": "France has a population of approximately 68 million.",
    "population of germany": "Germany has a population of approximately 84 million.",
}


def lookup_fact(topic: str) -> str:
    key = topic.strip().lower()
    return FACTS.get(key, f"No fact found for '{topic}'. Known topics: {', '.join(FACTS)}")


def delete_file(path: str) -> str:
    # Deliberately never actually deletes anything — the point of this
    # template is demonstrating that the breakpoint fires and pauses
    # execution BEFORE this function would ever run for real, not
    # building a real file-deletion tool.
    return f"(simulated) Deleted {path}"


TOOLS = [
    {
        "name": "lookup_fact",
        "description": "Look up a known fact by topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file by path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "lookup_fact":
        return lookup_fact(**tool_input), False
    if name == "delete_file":
        return delete_file(**tool_input), False
    return f"Unknown tool: {name}", True


class LoopState:
    """CONCEPT: the running state breakpoints check against — tracked
    across the whole session (not reset per turn), since a cost budget
    or a stuck-loop pattern is about cumulative/recent behavior, not any
    single call in isolation.
    """

    def __init__(self) -> None:
        self.total_tokens = 0
        self.recent_calls: list[tuple[str, str]] = []  # (tool_name, sorted-args-string), most recent last

    def record_usage(self, usage) -> None:
        self.total_tokens += usage.input_tokens + usage.output_tokens

    def record_call(self, tool_name: str, tool_input: dict) -> None:
        signature = (tool_name, str(sorted(tool_input.items())))
        self.recent_calls.append(signature)
        self.recent_calls = self.recent_calls[-PROGRESS_STALL_THRESHOLD:]

    def is_stalled(self) -> bool:
        if len(self.recent_calls) < PROGRESS_STALL_THRESHOLD:
            return False
        return len(set(self.recent_calls)) == 1  # every recent call was identical


def check_breakpoints(state: LoopState, tool_name: str, tool_input: dict) -> str | None:
    """CONCEPT: the check itself — run once per requested tool call,
    before that call is dispatched. Returns a human-readable reason if
    ANY breakpoint condition is met, or None if it's fine to proceed.
    Order matters only for which message is shown first; execution halts
    on the first match either way.
    """
    if state.total_tokens >= COST_BUDGET_TOKENS:
        return f"COST breakpoint: cumulative usage ({state.total_tokens} tokens) has reached the {COST_BUDGET_TOKENS}-token budget."

    if tool_name in ACTION_BREAKPOINTS:
        return f"ACTION breakpoint: '{tool_name}' is flagged as requiring a pause before it runs, regardless of arguments."

    # Check for a stall assuming this call is added to recent history.
    projected = state.recent_calls + [(tool_name, str(sorted(tool_input.items())))]
    projected = projected[-PROGRESS_STALL_THRESHOLD:]
    if len(projected) == PROGRESS_STALL_THRESHOLD and len(set(projected)) == 1:
        return f"PROGRESS breakpoint: the same call ({tool_name}({tool_input})) has been requested {PROGRESS_STALL_THRESHOLD} times in a row — the agent may be stuck."

    return None


def prompt_human(reason: str) -> bool:
    """Blocks on real terminal input, same as
    ../human_in_the_loop/human_in_the_loop.py's approval gate — but here
    triggered by a CONDITION, not by which tool was called.
    """
    print(f"\n  >>> BREAKPOINT HIT: {reason}")
    decision = input("      [c]ontinue this one call / [a]bort the task? ").strip().lower()
    return decision == "c"


def run_turn(messages: list[dict], state: LoopState) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        state.record_usage(response.usage)
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        aborted = False
        for block in response.content:
            if block.type == "tool_use":
                # CONCEPT: the gate — checked for EVERY requested call,
                # before dispatch, regardless of which tool it is.
                reason = check_breakpoints(state, block.name, block.input)
                if reason and not prompt_human(reason):
                    print("  [aborted by human at breakpoint]")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Task aborted by human at a breakpoint: {reason}",
                            "is_error": True,
                        }
                    )
                    aborted = True
                    continue

                state.record_call(block.name, block.input)
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
        if aborted:
            return


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Interrupts/breakpoints demo. Type 'exit' to quit.\n")
    print("Try: \"Delete the file old_report.txt\" (triggers the ACTION breakpoint)\n")

    messages: list[dict] = []
    state = LoopState()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages, state)


if __name__ == "__main__":
    main()
