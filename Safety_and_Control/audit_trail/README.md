# audit_trail

Audit Trail / Logging — recording every action and decision the agent makes for review and debugging, in a durable form a human is meant to read.

## audit_trail.py

A file-management assistant where every user message, assistant response, tool call, and tool result is written to an append-only JSONL log as it happens. Type `exit` to quit and see the session's full log replayed from disk.

### Concepts covered

- **Written for humans, never read back by the agent** — contrast with `../../Task_and_State_Management/checkpointing/checkpointing.py` (persisted so the *same agent* can resume) and `../../Memory/episodic_memory/episodic_memory.py` (persisted so the agent can *recall* it later). Nothing in this file loads the audit log back into a prompt — `print_audit_log` is the only reader, and it's meant for a human.
- **`AuditLog.record`** — structured (one JSON object per line: timestamp, sequence number, event type, payload) and append-only (opened in append mode, flushed immediately per entry) — verified that entries land with strictly increasing sequence numbers and that multiple sessions can share one file without cross-contamination, filterable by `session_id`.
- **Nothing is filtered or summarized** — every user message, assistant text block, tool call, and tool result is logged, because you can't debug what wasn't recorded. Verified with a full mocked tool round-trip that the log captures the complete event sequence in the correct order: `user_message → assistant_text → tool_call → tool_result → assistant_text`.
- **`print_audit_log` reads only from disk** — proving the log is a genuine independent record: it never touches in-memory session state, so it would work identically even if run by a separate process after the original one exited.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Safety_and_Control/audit_trail/audit_trail.py
```

Try:

```
You: Rename notes.txt to project_notes.txt
  [tool] rename_file({'old_name': 'notes.txt', 'new_name': 'project_notes.txt'})
  [result] Renamed 'notes.txt' to 'project_notes.txt'.
You: exit

=== Audit log for session 4f9a2c1b ===
  [  1] 2026-...  user_message: {'text': 'Rename notes.txt to project_notes.txt'}
  [  2] 2026-...  tool_call: {'tool_name': 'rename_file', 'tool_input': {...}}
  [  3] 2026-...  tool_result: {'tool_name': 'rename_file', 'result': "Renamed...", 'is_error': False}
  [  4] 2026-...  assistant_text: {'text': "Done — renamed..."}
  [  5] 2026-...  session_end: {}
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `AUDIT_LOG_PATH` — where the log is written (default: `audit_log.jsonl` next to the script)

### See also

- `../../Task_and_State_Management/checkpointing/README.md` — persistence the agent reads back to resume, contrasted with this template's human-only record
- `../../Memory/episodic_memory/README.md` — persistence the agent reads back to inform its own future behavior, also contrasted with this template's human-only record
