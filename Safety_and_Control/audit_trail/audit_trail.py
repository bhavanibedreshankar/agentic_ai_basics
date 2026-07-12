"""
CONCEPT: Audit Trail / Logging — recording every action and decision an
agent makes, in a durable, structured form, for a HUMAN to review or
debug later. The record is written FOR people outside the loop, not for
the agent itself.

That last point is what separates this from persistence elsewhere in
the repo that looks superficially similar:
  - `../../Task_and_State_Management/checkpointing/checkpointing.py`
    persists progress so the SAME AGENT can resume — the checkpoint is
    read back by the program, mutated, and eventually irrelevant once
    the task finishes.
  - `../../Memory/episodic_memory/episodic_memory.py` persists past
    interactions so the AGENT can recall them and change its OWN future
    behavior — it's read back and acted on, by the agent.
  - An audit log here is never read back by the agent AT ALL. It's
    written once, per event, and left alone — a human (or a separate
    review tool) is the only intended reader. Nothing in this file
    loads the log back into a prompt or a decision.

Two properties make a log actually useful for audit/debugging rather
than just "some printed lines":
  1. STRUCTURED, one JSON object per line (JSONL) — every entry has the
     same shape (timestamp, sequence number, event type, payload), so it
     can be filtered, grepped, or loaded into a table later, unlike
     free-text prose.
  2. APPEND-ONLY — the file is opened in append mode and each entry is
     written and flushed immediately as it happens, not batched and
     rewritten at the end. If the process crashes mid-task, the log up
     to that point is still intact and readable — the log doesn't
     depend on a clean shutdown to be trustworthy.

Every user message, assistant text, tool call, tool result, and error is
logged — nothing is summarized or filtered out, because you can't debug
what wasn't recorded. Use case: a small file-management agent, logged
end to end; `print_audit_log` at the end of a session replays the log
from disk to prove it's a complete, independent record of what happened.
Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = "You are a file-management assistant. Use list_files and rename_file to help the user organize files."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.jsonl"

FILES = ["draft_v1.txt", "notes.txt", "IMG_2024.jpg"]


def list_files() -> str:
    return "\n".join(FILES)


def rename_file(old_name: str, new_name: str) -> str:
    if old_name not in FILES:
        return f"Error: '{old_name}' not found."
    FILES[FILES.index(old_name)] = new_name
    return f"Renamed '{old_name}' to '{new_name}'."


TOOLS = [
    {
        "name": "list_files",
        "description": "List all files.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "rename_file",
        "description": "Rename a file.",
        "input_schema": {
            "type": "object",
            "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}},
            "required": ["old_name", "new_name"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "list_files":
        return list_files(), False
    if name == "rename_file":
        return rename_file(**tool_input), False
    return f"Unknown tool: {name}", True


class AuditLog:
    """CONCEPT: the audit mechanism itself. `_sequence` gives every entry
    in a session a strictly increasing order number independent of wall-
    clock time (useful when two events share a timestamp at the
    resolution logged); `session_id` lets entries from different runs
    share one log file without becoming ambiguous about which run an
    entry belongs to.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.session_id = uuid.uuid4().hex[:8]
        self._sequence = 0

    def record(self, event_type: str, payload: dict) -> None:
        self._sequence += 1
        entry = {
            "session_id": self.session_id,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event_type": event_type,
            **payload,
        }
        # Append-only, flushed immediately — this entry is durable the
        # moment record() returns, not batched for a later write.
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def print_audit_log(session_id: str) -> None:
    """CONCEPT: proving the log is a genuine independent record — this
    reads ONLY from disk, never from in-memory state, to reconstruct
    what happened in a session. If this function is all you have (the
    process that ran the session is long gone), it's still enough to see
    every decision and action in order.
    """
    if not AUDIT_LOG_PATH.exists():
        print("(no audit log yet)")
        return
    print(f"\n=== Audit log for session {session_id} ===")
    with AUDIT_LOG_PATH.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["session_id"] != session_id:
                continue
            print(f"  [{entry['sequence']:>3}] {entry['timestamp']}  {entry['event_type']}: "
                  f"{ {k: v for k, v in entry.items() if k not in ('session_id', 'sequence', 'timestamp', 'event_type')} }")


def run_turn(messages: list[dict], log: AuditLog) -> None:
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
                log.record("assistant_text", {"text": block.text})

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                log.record("tool_call", {"tool_name": block.name, "tool_input": block.input})
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [result] {result_text}")
                log.record("tool_result", {"tool_name": block.name, "result": result_text, "is_error": is_error})
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

    log = AuditLog(AUDIT_LOG_PATH)
    print(f"File assistant (audit trail demo) — session {log.session_id}. Type 'exit' to quit.\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            log.record("session_end", {})
            print_audit_log(log.session_id)
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        log.record("user_message", {"text": user_input})
        messages.append({"role": "user", "content": user_input})
        run_turn(messages, log)


if __name__ == "__main__":
    main()
