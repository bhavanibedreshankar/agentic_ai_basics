# adversarial_safety_testing

Red-teaming the agent with prompt injection attempts, jailbreak attempts, and malformed/hostile tool inputs to verify it refuses or degrades safely instead of measuring normal-case correctness.

## adversarial_safety_testing.py

A pytest-style test module pinning down `../../10_Safety_and_Control/guardrails/guardrails.py`'s `check_input_guardrail`/`check_action_guardrail` checks (reimplemented here) against a growing corpus of known attack strings and a corpus of merely-suspicious-looking benign strings, plus a new `check_tool_argument_safety` check for hostile raw argument values. Run with `pytest adversarial_safety_testing.py` or directly with `python3 adversarial_safety_testing.py`.

### Concepts covered

- **`ATTACK_CORPUS` / `test_attack_corpus_all_blocked`** — a permanent, growing list of known attack phrasings; every real attack ever found gets added here so the exact same trick can never silently start working again.
- **`BENIGN_CORPUS` / `test_benign_corpus_never_blocked`** — the false-positive check: a guardrail that blocks legitimate requests mentioning "instructions" or "system" is broken in the other direction, and only testing attacks would never reveal that.
- **`test_indirect_injection_in_tool_result_detected`** — proves the injection scanner must also run over TOOL RESULTS, not just user text; indirect prompt injection often arrives hidden in retrieved content (a web page, a note field) rather than typed by the user.
- **`check_tool_argument_safety`** — validates raw argument values (a numeric-only order ID pattern, refund amount type/sign checks) that go beyond `guardrails.py`'s refund-cap-only action guardrail; see `test_hostile_order_id_rejected` and `test_refund_amount_type_confusion_rejected`.
- **`_run_tests_manually()`** — same dependency-free runner as `../tool_unit_testing/tool_unit_testing.py`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 14_Agent_Testing/adversarial_safety_testing/adversarial_safety_testing.py
# or
pytest 14_Agent_Testing/adversarial_safety_testing/adversarial_safety_testing.py
```

Try:

```
  [PASS] test_attack_corpus_all_blocked
  [PASS] test_benign_corpus_never_blocked
  [PASS] test_indirect_injection_in_tool_result_detected
  [PASS] test_hostile_order_id_rejected
  ...

9 passed, 0 failed
```

### Configuration

- `INJECTION_PATTERNS`, `MAX_REFUND_AMOUNT`, `SAFE_ORDER_ID_PATTERN` — the policy constants under test
- No `ANTHROPIC_API_KEY` needed — every check here is deterministic Python, no model call

### See also

- `../../10_Safety_and_Control/guardrails/README.md` — the guardrail checks this file's corpus tests are written against
- `../tool_unit_testing/README.md` — the same "test the error cases, not just the happy path" discipline, applied to normal invalid input instead of deliberately hostile input
- `../test_suite_pyramid/README.md` — where this fast, deterministic tier fits relative to the expensive LLM-judge tier
