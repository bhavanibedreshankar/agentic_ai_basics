# guardrails

Guardrails — rules or classifiers that prevent the agent from taking harmful or out-of-scope actions, enforced automatically with no human in the loop for the common case.

## guardrails.py

A customer support agent (account lookup, refunds) guarded by two independent layers: an input classifier that screens for prompt-injection patterns before any model call, and an action policy that blocks over-cap refund requests before they execute. Type `exit` to quit.

### Concepts covered

- **Automatic rejection, not a human pause** — contrast with `../../Execution_Loops/interrupts_breakpoints/interrupts_breakpoints.py`, which *pauses* for a human decision when a condition fires. Guardrails here reject outright, no human involved, with an explanation the model can act on.
- **Content rules, not state legality** — contrast with `../../Task_and_State_Management/state_machine/state_machine.py`, which checks whether a *transition* is legal from the current state. `check_action_guardrail` checks whether a call's *content* (a refund amount) is within policy, independent of any state machine.
- **Two enforcement points** — `check_input_guardrail` runs on raw user text *before* it's ever sent to the API (a rejected input costs zero tokens and never reaches the model); `check_action_guardrail` runs on a tool call's arguments *before* dispatch, the same placement pattern `state_machine.py` uses for legality checks.
- **Verified both layers are real, not cosmetic** — tested that injection-pattern input is blocked pre-API, benign input isn't false-flagged, and a full loop integration proves an over-cap refund request never reaches `execute_tool`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Safety_and_Control/guardrails/guardrails.py
```

Try:

```
You: Refund account A100 $500
  [GUARDRAIL BLOCKED] issue_refund({'account_id': 'A100', 'amount': 500}) — Refund of $500.00 exceeds the maximum allowed refund of $100.00.

Claude: I can't process a refund that large — the maximum I can issue is $100...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `INJECTION_PATTERNS` — the input guardrail's regex list
- `MAX_REFUND_AMOUNT` — the action guardrail's policy threshold

### See also

- `../../Execution_Loops/interrupts_breakpoints/README.md` — the human-pause alternative to this template's automatic rejection
- `../permission_scoping/README.md` — restricting which tools exist at all, rather than checking a call's content after the fact
