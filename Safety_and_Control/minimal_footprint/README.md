# minimal_footprint

Minimal Footprint Principle — agents should request only necessary permissions and avoid side effects beyond the task, re-evaluated fresh for every request rather than fixed in advance.

## minimal_footprint.py

A workspace assistant (search notes, edit a note, send a digest email) where the tool grant is computed per-task from a local, deterministic keyword classifier — not a fixed role. Type `exit` to quit.

### Concepts covered

- **Per-task, not per-role** — contrast with `../permission_scoping/permission_scoping.py`: that template grants a fixed tool set for an entire session based on a role, decided once. `select_tools_for_task` re-derives the grant for *every individual request*, and can be narrower than even a role's full allowance would be — a read-only-shaped request gets exactly one tool, regardless of what else the session might be permitted to do.
- **A local classifier, deliberately not another LLM call** — `infer_required_capabilities` is plain keyword matching against `CAPABILITY_TRIGGERS`. The docstring is explicit about why: asking a second model "what does this need" would add its own cost and failure surface on top of the very call it's supposed to reduce exposure for.
- **Capabilities as a subset check** — each tool declares the capabilities it needs (`{"read"}`, `{"read", "write"}`, `{"read", "external_communication"}`); a tool is granted only when its capability set is a subset of what the task was inferred to need. Verified directly: a write-shaped task grants `edit_note` but *not* `send_digest_email`, and an email-shaped task grants `send_digest_email` but *not* `edit_note` — cross-contamination between capability types never happens.
- **Defense in depth** — `execute_tool` re-checks the per-task grant list even when called directly, same pattern as `../permission_scoping/permission_scoping.py`'s role check.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Safety_and_Control/minimal_footprint/minimal_footprint.py
```

Try three different requests in a row and watch the granted tool set change each time:

```
Task: What's on my todo list?
  [inferred capabilities: ['read']]
  [granted tools: ['search_notes'] (of 3 total available)]

Task: Update my todo list to add 'call the bank'
  [inferred capabilities: ['read', 'write']]
  [granted tools: ['search_notes', 'edit_note'] (of 3 total available)]

Task: Email a digest to sam@example.com
  [inferred capabilities: ['external_communication', 'read']]
  [granted tools: ['search_notes', 'send_digest_email'] (of 3 total available)]
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `TOOL_REGISTRY` — each tool's declared capabilities; add a tool by tagging it with the capabilities it actually needs
- `CAPABILITY_TRIGGERS` — the keyword sets the local classifier matches against

### See also

- `../permission_scoping/README.md` — the coarser, role-based version of this same least-privilege idea
- `../../Agent_Frameworks_and_Patterns/tool_registry/README.md` — the general tool-registry pattern this template's `TOOL_REGISTRY` builds on, extended with capability tags
