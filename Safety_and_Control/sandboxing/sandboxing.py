"""
CONCEPT: Sandboxing — isolating agent execution, especially code, so
that even a fully unrestricted or malicious command can't affect
anything beyond a contained boundary.

This is a genuinely different mechanism from the file-path protection in
`../../Tools_and_Actions/file_io_tools/file_io_tools.py`: that template
validates a single argument (a path) before a read/write. This template
sandboxes ARBITRARY COMMAND EXECUTION, which has a much bigger attack
surface — shell metacharacters, command chaining (`;`, `&&`, `|`),
argument injection, environment variables, resource exhaustion. A single
path check doesn't cover any of that.

It's also different from `../../Tools_and_Actions/code_interpreter/code_interpreter.py`:
that template hands code execution off entirely to Anthropic's own
hosted sandbox — there's no sandboxing CODE in that file, because
Anthropic's infrastructure already does it. This template is what you'd
build if you're hosting the execution environment YOURSELF — a
`run_command` tool for a coding agent, sandboxed with four independent,
STACKED layers of restriction, each of which closes off a different
class of attack:

  1. ALLOWLIST — only explicitly permitted binaries can run at all;
     everything else is rejected by name before anything executes.
  2. NO SHELL — commands are parsed with `shlex.split` and run via
     `subprocess.run(args, shell=False)`, never through a shell
     interpreter. This is what actually defeats `;`, `&&`, `|`, backticks,
     and `$()` — they're just inert characters in an argument, not shell
     syntax, because there's no shell present to interpret them.
  3. TIMEOUT — a command that hangs (or is deliberately designed to)
     is killed after a fixed wall-clock limit rather than blocking the
     agent (or a human) forever.
  4. CONFINED WORKING DIRECTORY — every command runs with `cwd` pinned to
     a sandbox directory, the same idea as file_io_tools's SANDBOX_ROOT,
     applied to command execution instead of file paths.

Use case: a coding assistant with a `run_command` tool sandboxed this
way. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a coding assistant with a sandboxed run_command tool. Only "
    "a small set of safe commands are available — if a command is "
    "rejected, explain that to the user rather than trying variations to "
    "get around it."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SANDBOX_DIR = (Path(__file__).parent / "sandbox").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)

# CONCEPT: layer 1 — the allowlist. Only these binaries can run, full
# stop. This is the FIRST check, before any of the others, since there's
# no point parsing or sandboxing a command that shouldn't run at all.
ALLOWED_COMMANDS = {"echo", "ls", "cat", "wc", "date", "pwd"}

COMMAND_TIMEOUT_SECONDS = 5


class SandboxViolation(Exception):
    pass


def run_sandboxed_command(command: str) -> str:
    """CONCEPT: all four layers, stacked, in order. Any one of them
    failing stops execution before the command runs for real.
    """
    # Layer 2 (parse, don't interpret): shlex.split parses the string
    # into argv-style tokens the way a shell WOULD, but without ever
    # invoking a shell to interpret metacharacters. `ls; rm -rf /` splits
    # into ['ls;', 'rm', '-rf', '/'] here — a literal, meaningless
    # program name ("ls;"), not a chained command — because there's no
    # shell present to treat ';' as a separator.
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise SandboxViolation(f"Could not parse command: {exc}")

    if not parts:
        raise SandboxViolation("Empty command.")

    # Layer 1: the allowlist check, on the actual program name that will
    # run — not on the raw string, so it can't be fooled by prefixing or
    # embedding a chain the shell would otherwise have split apart.
    program = parts[0]
    if program not in ALLOWED_COMMANDS:
        raise SandboxViolation(f"'{program}' is not an allowed command. Allowed: {sorted(ALLOWED_COMMANDS)}")

    try:
        result = subprocess.run(
            parts,
            shell=False,               # Layer 2 — never invoke a shell
            cwd=SANDBOX_DIR,            # Layer 4 — confined working directory
            timeout=COMMAND_TIMEOUT_SECONDS,  # Layer 3 — bounded execution time
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise SandboxViolation(f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s.")
    except FileNotFoundError:
        raise SandboxViolation(f"'{program}' is allowlisted but not installed in this environment.")

    output = result.stdout
    if result.returncode != 0:
        output += f"\n(exit code {result.returncode}, stderr: {result.stderr.strip()})"
    return output.strip() or "(no output)"


TOOLS = [
    {
        "name": "run_command",
        "description": f"Run a shell command in a sandboxed environment. Only these commands are allowed: {sorted(ALLOWED_COMMANDS)}.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run, e.g. 'ls -la'"}},
            "required": ["command"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "run_command":
        try:
            return run_sandboxed_command(tool_input["command"]), False
        except SandboxViolation as exc:
            return f"Sandbox rejected this command: {exc}", True
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

    print(f"Sandboxed command assistant — allowed commands: {sorted(ALLOWED_COMMANDS)}. Type 'exit' to quit.\n")
    print("Try: \"Run rm -rf /\" (rejected — not on the allowlist)\n")

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
