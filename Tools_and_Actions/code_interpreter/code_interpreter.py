"""
CONCEPT: Code Interpreter — a sandboxed Python/shell execution environment
the agent can use to run code, unlike EVERY other tool built so far in
this repo.

Every custom tool up to this point (`../../Core_Architecture/tool_use/`, `../../Agent_Frameworks_and_Patterns/tool_registry/`,
`../../Memory/external_memory/`, etc.) is CLIENT-SIDE: Claude sends a
tool_use request, YOUR code runs the actual function, and you send the
result back as a tool_result. Code execution is different — it's a
SERVER-SIDE tool. Declare `{"type": "code_execution_20260521", "name":
"code_execution"}` in `tools` and Claude runs the code itself, in a
sandboxed container on Anthropic's infrastructure. There is no
execute_tool() dispatch function anywhere in this file, and no
tool_result to send back — that's not a simplification, it's the actual
mechanic. The whole request/response is one round trip; the code and its
output just show up as extra content blocks in the response.

This also means there's nothing to mock here, unlike most of this repo's
tools — this template calls the real API feature directly. The container
(1 CPU, 5 GiB RAM, no internet access, Python 3.11 with pandas/numpy/
matplotlib preinstalled) persists for 30 days and can be reused across
requests by passing its `container` id back on the next call, which this
template does automatically within a session — so a later question can
reference a variable or file a previous turn created.

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
    "You are a data analysis assistant with access to a Python code "
    "execution sandbox. Use it to compute exact answers rather than "
    "estimating — for any question involving calculation, statistics, or "
    "data manipulation, write and run code instead of reasoning it out "
    "in your head."
)

# CONCEPT: the tool declaration is the ENTIRE integration. No handler
# function, no input_schema (this is an Anthropic-defined tool, not a
# custom one — see ../file_io_tools/ for the client-side equivalent of
# that distinction).
TOOLS = [
    {"type": "code_execution_20260521", "name": "code_execution"},
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def print_response_content(response) -> None:
    """Walk the response and print each block type distinctly, so the
    code Claude wrote and the output it produced are both visible —
    not just the final text summary.
    """
    for block in response.content:
        if block.type == "text":
            print(f"\nClaude: {block.text}")
        elif block.type == "server_tool_use" and block.name == "code_execution":
            code = block.input.get("code", "")
            print(f"\n  [running code]\n{code}")
        elif block.type == "bash_code_execution_tool_result":
            result = block.content
            if getattr(result, "type", None) == "bash_code_execution_result":
                if result.stdout:
                    print(f"  [stdout] {result.stdout.strip()}")
                if result.stderr:
                    print(f"  [stderr] {result.stderr.strip()}")
            else:
                print(f"  [tool error] {result}")


def run_turn(messages: list[dict], container_id: str | None) -> str | None:
    """Send one turn. Returns the container id from the response, so the
    caller can pass it back on the NEXT turn and reuse the same sandbox
    (same installed packages, same files on disk) instead of starting
    fresh every time.
    """
    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": TOOLS,
        "output_config": {"effort": EFFORT},
        "messages": messages,
    }
    if container_id:
        kwargs["container"] = container_id

    response = client.messages.create(**kwargs)
    messages.append({"role": "assistant", "content": response.content})
    print_response_content(response)

    # response.container is only set when code execution actually ran.
    new_container_id = response.container.id if response.container else container_id
    return new_container_id


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Code interpreter demo. Type 'exit' to quit.\n")
    print(
        "Try: \"Calculate the standard deviation of [4, 8, 15, 16, 23, 42] "
        "and tell me which values are more than 1 std dev from the mean.\"\n"
    )

    messages: list[dict] = []
    container_id: str | None = None

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        container_id = run_turn(messages, container_id)
        print()


if __name__ == "__main__":
    main()
