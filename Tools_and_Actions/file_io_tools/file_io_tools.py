"""
CONCEPT: File I/O Tools — letting the agent read, write, and edit files
on disk, the core capability behind every coding agent.

Unlike `../code_interpreter/` and `../web_search/`, this is a CLIENT-SIDE
Anthropic-defined tool: Claude requests an action (view a file, create
one, replace a string, insert a line), but YOUR code performs it. It's
"Anthropic-defined" rather than a fully custom tool like everything in
`../../tool_use/` or `../../tool_registry/` because the schema and command
set are built into the model — declare it by type and name only,
`{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}`,
with NO input_schema (the model already knows the shape).

The part that actually matters here is security, not mechanics. `path`
in every command is MODEL-SUPPLIED, which means it's untrusted input —
even though it comes from Claude rather than directly from the end user,
a manipulated conversation (prompt injection from a tool result, a
malicious document Claude was asked to summarize, etc.) could cause it to
request `path: "/etc/passwd"` or `path: "../../../.ssh/id_rsa"`.
`_resolve_safe_path` below is what stops that: every path is resolved to
its canonical absolute form and checked against SANDBOX_ROOT before any
file operation runs, and anything that would escape the sandbox is
rejected as a tool error instead of executed. This is the pattern every
real coding agent needs, not an optional extra — see the SECURITY note
in `../../tool_use/README.md`'s underlying tool-use concepts for the
general version of this rule.

All operations are confined to a `sandbox/` directory next to this
script. Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a coding assistant with access to file editing tools, "
    "scoped to a sandbox directory. Use them to view, create, and edit "
    "files as needed to complete the user's request."
)

SANDBOX_ROOT = (Path(__file__).parent / "sandbox").resolve()
SANDBOX_ROOT.mkdir(exist_ok=True)

TOOLS = [
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


class UnsafePathError(Exception):
    pass


def _resolve_safe_path(path_str: str) -> Path:
    """CONCEPT: the security boundary. Resolve the model-supplied path
    (which may be relative, may contain '..', may be absolute) against
    the sandbox root, canonicalize it (following any symlinks, collapsing
    '..' segments), and verify the result is still inside SANDBOX_ROOT.
    Anything that resolves outside — a traversal attempt, an absolute
    path elsewhere on disk — is rejected before touching the filesystem.
    """
    # A leading '/' from the model means "root of the sandbox" here, not
    # the real filesystem root — strip it so it joins rather than escapes.
    candidate = (SANDBOX_ROOT / path_str.lstrip("/")).resolve()
    if not candidate.is_relative_to(SANDBOX_ROOT):
        raise UnsafePathError(f"Path '{path_str}' resolves outside the sandbox — refusing.")
    return candidate


def execute_text_editor(tool_input: dict) -> tuple[str, bool]:
    """Dispatch a text_editor command. Every branch resolves its path
    through _resolve_safe_path FIRST, before any read/write happens.
    """
    command = tool_input.get("command")
    try:
        if command == "view":
            path = _resolve_safe_path(tool_input["path"])
            if path.is_dir():
                entries = sorted(p.name for p in path.iterdir())
                return f"Directory listing for {tool_input['path']}:\n" + "\n".join(entries), False
            if not path.exists():
                return f"Error: {tool_input['path']} does not exist", True
            lines = path.read_text().splitlines()
            view_range = tool_input.get("view_range")
            if view_range:
                start, end = view_range
                lines = lines[start - 1 : end if end != -1 else None]
                start_line = start
            else:
                start_line = 1
            numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=start_line))
            return numbered, False

        if command == "create":
            path = _resolve_safe_path(tool_input["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tool_input["file_text"])
            return f"Created {tool_input['path']}", False

        if command == "str_replace":
            path = _resolve_safe_path(tool_input["path"])
            content = path.read_text()
            old_str, new_str = tool_input["old_str"], tool_input["new_str"]
            count = content.count(old_str)
            if count != 1:
                return f"Error: old_str found {count} times in {tool_input['path']}, expected exactly 1", True
            path.write_text(content.replace(old_str, new_str, 1))
            return f"Replaced text in {tool_input['path']}", False

        if command == "insert":
            path = _resolve_safe_path(tool_input["path"])
            lines = path.read_text().splitlines()
            insert_line = tool_input["insert_line"]
            lines.insert(insert_line, tool_input["insert_text"])
            path.write_text("\n".join(lines) + "\n")
            return f"Inserted text into {tool_input['path']} after line {insert_line}", False

        return f"Unknown command: {command}", True

    except UnsafePathError as exc:
        return f"Security error: {exc}", True
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
                result_text, is_error = execute_text_editor(block.input)
                print(f"  [result] {result_text[:200]}")
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

    print(f"File editing assistant — sandboxed to {SANDBOX_ROOT}")
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
