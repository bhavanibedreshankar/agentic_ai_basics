"""
CONCEPT: Orchestrator — a high-level agent that breaks a goal into
subtasks and delegates each one to a specialized sub-agent, rather than
doing the work itself.

Contrast with ../../Planning_and_Reasoning/plan_and_execute/: that
template generates a plan of STEPS and executes every one of them with
the SAME generic "execute this step" call — every step gets an
identical, general-purpose executor. An orchestrator is different: each
subtask goes to a DIFFERENT, SPECIALIZED agent — a researcher, a writer,
an editor — each with its own system prompt (its own persona and
expertise) rather than one interchangeable executor.

The mechanic: the orchestrator sees each sub-agent as a TOOL —
delegate_to_researcher, delegate_to_writer, delegate_to_editor. Calling
one of these dispatches to a completely separate API call under a
different system prompt, and the sub-agent's response comes back as a
tool_result, the exact same shape as any other tool in this repo. The
orchestrator decides which specialist to call and in what order based on
the request — there's no fixed sequence hardcoded anywhere.

Each delegate_to_X function here is a single, non-looping call — see
../worker_agent/ for what it looks like when a sub-agent is expanded into
a full agent with its own tools and its own internal tool-calling loop,
rather than a single-shot specialized call.

Use case: a content creation orchestrator managing three specialists —
researcher, writer, editor. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are an orchestrator managing a small content team: a researcher, "
    "a writer, and an editor. Break the user's request into subtasks and "
    "delegate each one to the right specialist using the tools available "
    "— don't research, write, or edit anything yourself. Combine the "
    "specialists' outputs into your final response to the user."
)


# ---------------------------------------------------------------------------
# The specialists. Each is a single focused API call under its own system
# prompt — the same "narrow, single-purpose call" idea as
# ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's steps, but invoked
# dynamically by the orchestrator's own decisions rather than in a fixed
# order the developer wrote in advance.
# ---------------------------------------------------------------------------
def delegate_to_researcher(topic: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a research specialist. Given a topic, provide 3-4 concise, factual bullet points about it.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": topic}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def delegate_to_writer(brief: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a writing specialist. Given research notes and a brief, write a short, engaging paragraph.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": brief}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def delegate_to_editor(draft: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are an editing specialist. Tighten and polish the given draft — fix grammar, cut redundancy, improve flow. Return only the edited text.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": draft}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


ORCHESTRATOR_TOOLS = [
    {
        "name": "delegate_to_researcher",
        "description": "Delegate to the research specialist to gather key facts about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "The topic to research"}},
            "required": ["topic"],
        },
    },
    {
        "name": "delegate_to_writer",
        "description": "Delegate to the writing specialist to draft content from a brief or research notes.",
        "input_schema": {
            "type": "object",
            "properties": {"brief": {"type": "string", "description": "What to write about, including any research notes to draw on"}},
            "required": ["brief"],
        },
    },
    {
        "name": "delegate_to_editor",
        "description": "Delegate to the editing specialist to polish an existing draft.",
        "input_schema": {
            "type": "object",
            "properties": {"draft": {"type": "string", "description": "The draft text to edit"}},
            "required": ["draft"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "delegate_to_researcher":
        return delegate_to_researcher(**tool_input), False
    if name == "delegate_to_writer":
        return delegate_to_writer(**tool_input), False
    if name == "delegate_to_editor":
        return delegate_to_editor(**tool_input), False
    return f"Unknown specialist: {name}", True


def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=ORCHESTRATOR_TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nOrchestrator: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                specialist = block.name.replace("delegate_to_", "")
                print(f"  [delegating to {specialist}] {block.input}")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [{specialist} returned] {result_text[:150]}...")
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

    print("Content team orchestrator. Type 'exit' to quit.\n")
    print("Try: \"Write a short paragraph about the history of coffee, researched and polished.\"\n")

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
