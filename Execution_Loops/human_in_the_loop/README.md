# human_in_the_loop

Human-in-the-Loop (HITL) — a human approves or corrects an agent's actions at key checkpoints, before they execute.

## human_in_the_loop.py

An email assistant that can freely search a mock inbox, but must get human approval before actually sending anything — the human can approve, reject with feedback, or edit the message first. Type `exit` to quit.

### Concepts covered

- **One gate inserted into the ordinary tool loop** — contrast with `../../Tools_and_Actions/tool_use/basic_agentic_tools.py`, which dispatches a requested tool immediately. Here, `run_turn` checks `REQUIRES_APPROVAL` before dispatch; everything else in the loop is identical.
- **The policy is reversibility-based** — `REQUIRES_APPROVAL = {"send_email"}`: reading (`search_inbox`) is auto-approved, anything that changes state outside the program requires a human. The module docstring is explicit that this line is a design decision, not a technical one.
- **Three real outcomes, not just yes/no** — `request_human_approval` returns approve-as-is, reject-with-feedback (the tool never runs; the feedback becomes the `tool_result` so Claude can react), or edit-then-approve (specific fields changed before the real call runs).
- **Verified that rejection is real, not cosmetic** — tested with a full `run_turn` integration (mocked API response + mocked human input) proving `execute_tool` is *never* invoked when a human rejects — not just that the printed message looks right.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Execution_Loops/human_in_the_loop/human_in_the_loop.py
```

Try:

```
You: Reply to Sara agreeing to move the budget review to Tuesday.

  >>> APPROVAL NEEDED: send_email({'to': 'sara@example.com', 'subject': 'Re: Q3 budget review', 'body': '...'})
      [a]pprove / [r]eject / [e]dit? a
  [tool] send_email({...})
  [result] Email sent to sara@example.com — subject: 'Re: Q3 budget review'
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `REQUIRES_APPROVAL` — the set of tool names that trigger the gate; add or remove tool names to change the policy
- `INBOX` — the mock data `search_inbox` searches over

### See also

- `../interrupts_breakpoints/README.md` — a gate keyed on *conditions* over running state instead of a fixed tool-name policy
- `../../Tools_and_Actions/tool_use/README.md` — the underlying tool-calling loop this template adds one approval check to
