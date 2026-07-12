# agent_handoff

Agent Handoff — passing control and context from one agent to another in a pipeline, permanently, rather than delegating and getting a result back.

## agent_handoff.py

A customer support triage agent that routes to a billing, technical, or general specialist — whichever is chosen then handles the rest of the conversation directly, with no further involvement from triage. Type `exit` to end the conversation.

### Concepts covered

- **Transfer, not delegation** — contrast with `../orchestrator/`: there, the orchestrator calls a specialist, gets a result back as a `tool_result`, and immediately regains control. Here, calling `transfer_to_billing` doesn't return a result at all — it flips `current_agent_name`, and control never comes back to triage.
- **The handoff happens mid-turn** — `run_turn`'s loop re-reads `AGENTS[current_agent_name]` at the top of every iteration, so the very next API call after a transfer — even within the same user turn — already uses the new agent's system prompt and tools. Verified in testing: triage hands off, and the *second* API call in the same `run_turn()` invocation is answered by billing, not triage.
- **State persists across turns** — `current_agent_name` is module-level, so once handed off, every *subsequent* user message goes straight to the specialist too, without re-running triage.
- **One-directional, on purpose** — specialists here have no transfer tools of their own, so handoff only flows one way. A natural extension: give specialists a `transfer_to_triage` tool for genuinely bidirectional routing.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Multi_Agent_Systems/agent_handoff/agent_handoff.py
```

Try:

```
[triage] You: I was charged twice for my subscription this month.
  [handoff: triage -> billing]

[billing] I can help with that — let's look into the duplicate charge...

[billing] You: ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `AGENTS` — each agent's system prompt and tools; add a new specialist by adding an entry here and a matching `transfer_to_X` tool in `TRANSFER_TOOLS`

### See also

- `../orchestrator/README.md` — the contrasting pattern where the delegator never gives up control
- `../worker_agent/README.md` — what a specialist could look like if it needed its own tools, rather than the plain personas used here
