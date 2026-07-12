# Safety_and_Control

Five mechanisms for keeping an agent's actions bounded and reviewable — restricting what it's even told exists, checking what it asks to do, isolating what actually runs, and recording all of it afterward.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`permission_scoping/`](permission_scoping/README.md) | Which tools a session even knows about, decided per role, before anything is requested |
| 2 | [`minimal_footprint/`](minimal_footprint/README.md) | The same idea taken further: a tool grant re-derived fresh per *task*, narrower than any fixed role |
| 3 | [`guardrails/`](guardrails/README.md) | Automatically rejecting unsafe input or out-of-policy tool calls, with no human involved for the common case |
| 4 | [`sandboxing/`](sandboxing/README.md) | Isolating what actually executes, so even an unrestricted or malicious command can't reach beyond a contained boundary |
| 5 | [`audit_trail/`](audit_trail/README.md) | Recording every action and decision to a durable, structured log a human can review afterward |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Safety_and_Control/guardrails/guardrails.py
```

## How these relate to each other

| | What it restricts | When it acts | Enforcement point |
|---|---|---|---|
| `permission_scoping/` | Which tools *exist* for a session | Once, at session start (per role) | The `tools=` list itself — an out-of-scope tool is never declared |
| `minimal_footprint/` | Which tools *exist* for a task | Fresh, on every request | Same as scoping, but re-derived per task instead of fixed per role |
| `guardrails/` | Specific inputs and specific tool calls | Before an API call (input) or before dispatch (action) | A content/policy check, independent of what's declared |
| `sandboxing/` | What a permitted action can actually *do* | While a permitted command runs | The execution environment itself (allowlist, no shell, timeout, confined cwd) |
| `audit_trail/` | Nothing — it doesn't restrict anything | After every event, unconditionally | A durable log, for human review, not enforcement |

The first four narrow *before* something happens, in increasingly late stages — scoping and footprint decide what's even offered, guardrails check what's requested, sandboxing contains what actually runs. `audit_trail/` is the odd one out: it doesn't prevent anything at all, it just makes sure that whatever *did* happen — permitted, blocked, or sandboxed — is reconstructable afterward. A production agent typically layers several of these together: a scoped or minimal-footprint tool set, guardrails on top of the tools that remain, sandboxed execution for anything code- or command-shaped, and an audit trail running underneath all of it.
