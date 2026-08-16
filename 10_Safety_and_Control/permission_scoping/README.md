# permission_scoping

Permission Scoping — limiting which tools or resources an agent can access based on the task context, by never declaring out-of-scope tools to the model at all.

## permission_scoping.py

A support-ticket assistant with three roles (`read_only`, `agent`, `admin`), each with a genuinely different `tools=` list built from the same underlying tool implementations. Type `exit` to quit.

### Concepts covered

- **Scoping the declaration, not the call** — contrast with `../guardrails/guardrails.py`: guardrails check a call's *content* after the model has already decided to request it — the tool is fully declared, some calls to it are rejected. Here, `build_tools_for_role` constructs a *different tool list per role*, so a `read_only` session's `tools=` never contains `delete_ticket`'s schema at all — there's no "the model asked and got rejected" moment, because it was never told the tool exists.
- **`ROLE_TOOLS` as a nested superset chain** — `admin`'s list is built by extending `agent`'s at import time (`ROLE_TOOLS["agent"] + ["delete_ticket"]`), not duplicated by hand — verified as a genuine strict-superset chain (`read_only ⊂ agent ⊂ admin`).
- **The tool schema is genuinely absent, not just hidden** — verified directly: `build_tools_for_role('read_only')` returns exactly 2 tool definitions, and `delete_ticket`'s schema is not among them, by construction.
- **Defense in depth** — `execute_tool` re-checks the role against `ROLE_TOOLS` even when called directly, in case something else in a larger system bypasses the scoped `tools=` list — verified that a `read_only` role is rejected from `delete_ticket` even when the call is made outside the normal loop.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 10_Safety_and_Control/permission_scoping/permission_scoping.py
```

Try running it three times with different roles and the same request ("delete ticket T1") to see the tool genuinely isn't available except under `admin`:

```
Choose a role (read_only/agent/admin): read_only

Running as 'read_only' — available tools: ['search_kb', 'view_ticket']

You: Delete ticket T1
Claude: I don't have a tool to delete tickets — I can only search and view them...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../01_Core_Architecture/basics/README.md`
- `ROLE_TOOLS` — the scoping table itself; add a role or extend an existing one's tool list here
- `SYSTEM_PROMPTS` — the per-role persona instructions

### See also

- `../guardrails/README.md` — checking a call's content after the model decides to make it, rather than restricting which tools exist beforehand
- `../minimal_footprint/README.md` — the same idea taken further: a tool grant computed per-*task*, not just per-role
