# supervisor_pattern

Supervisor Pattern — an orchestrator that monitors sub-agents, retries on failure, and validates outputs before trusting them.

## supervisor_pattern.py

A supervisor delegating structured data extraction (name, email, amount owed) from a support ticket to a worker, validating the worker's JSON output, and retrying with specific feedback when validation fails. Type `exit` to quit.

### Concepts covered

- **Deterministic validation, not another model's opinion** — contrast with `../../Planning_and_Reasoning/self_reflection/self_reflection.py`: that template's critic is itself an LLM call judging subjective quality. `validate_output` here is plain Python — parse JSON, check required fields, sanity-check the email — cheap, fast, and fully offline-testable, because there's nothing subjective to judge.
- **Retrying a *different* agent with feedback** — contrast with self-reflection's "one agent revises its own output": here the supervisor and the worker are separate roles, and each retry calls the worker again with a specific, actionable correction (`"Missing required field(s): amount, email."`) rather than the worker critiquing itself.
- **A hard stop, not infinite retries** — `MAX_RETRIES` bounds the loop; `supervise()` returns `None` and reports failure explicitly rather than looping forever on a worker that can't produce valid output.
- **Contrast with `../orchestrator/`** — that template trusts whatever a specialist returns; this one is the "trust but verify" version of the same delegation idea.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Multi_Agent_Systems/supervisor_pattern/supervisor_pattern.py
```

Try:

```
Ticket text: Hi, this is Jane Smith, jane.smith@example.com. I was billed $149.99 for a plan I cancelled last month.

[attempt 1/3] delegating to worker...
  worker output: {"name": "Jane Smith", "email": "jane.smith@example.com", "amount": 149.99}
  [supervisor validation] Valid.

=== Validated extraction ===
{
  "name": "Jane Smith",
  "email": "jane.smith@example.com",
  "amount": 149.99
}
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `MAX_RETRIES` — see `../../basics/README.md` for the first three; `MAX_RETRIES` caps retry attempts (default: `3`)
- `WORKER_SYSTEM_PROMPT` — what the worker is asked to extract and in what format
- `validate_output` — the validation criteria; extend with more checks (field types, value ranges) as needed

### See also

- `../orchestrator/README.md` — the trusting-by-default delegation pattern this template adds validation on top of
- `../../Planning_and_Reasoning/self_reflection/README.md` — the model-judged alternative for when criteria are subjective rather than checkable in code
