# tool_unit_testing

Testing individual tool functions in isolation, with deterministic inputs/outputs and no LLM call, including error/edge cases and mocked side effects.

## tool_unit_testing.py

A pytest-style test module (not a chat demo) covering the same order-lookup/refund tool domain as `../../Benchmarking/trace_evaluation/trace_evaluation.py`, but rewritten so every tool takes its data store as an explicit argument instead of reading a module global. Run with `pytest tool_unit_testing.py` or directly with `python3 tool_unit_testing.py`.

### Concepts covered

- **`fresh_orders()`** — a fixture function called fresh at the start of every test so state never leaks between tests; contrast with `../../Core_Architecture/tool_use/basic_agentic_tools.py`'s module-level `TASKS_FILE`, which is shared and stateful across the whole program.
- **`lookup_order` / `check_refund_eligibility` / `issue_refund`** — dependency-injected versions of the same tools in `../../Benchmarking/trace_evaluation/trace_evaluation.py`, now taking `orders: dict` as a parameter so tests can control starting state precisely.
- **`test_check_refund_eligibility_boundary_day`** — a boundary test proving the refund window is inclusive (`<=`), the kind of off-by-one bug a happy-path-only suite never catches.
- **`test_issue_refund_rejects_negative_amount` / `_rejects_zero_amount` / `_rejects_amount_over_price`** — error-case coverage: a tool should validate its own input rather than trusting the caller (see `../adversarial_safety_testing/README.md` for the same idea against deliberately hostile input).
- **`_run_tests_manually()`** — a dependency-free test runner so the file works with plain `python3` even without pytest installed, while remaining fully pytest-discoverable as-is.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 Agent_Testing/tool_unit_testing/tool_unit_testing.py
# or
pytest Agent_Testing/tool_unit_testing/tool_unit_testing.py
```

Try:

```
  [PASS] test_check_refund_eligibility_boundary_day
  [PASS] test_check_refund_eligibility_outside_window
  [PASS] test_issue_refund_rejects_negative_amount
  [PASS] test_lookup_order_not_found_raises
  ...

10 passed, 0 failed
```

### Configuration

- `REFUND_WINDOW_DAYS` — the eligibility cutoff used by `check_refund_eligibility`
- No `ANTHROPIC_API_KEY` needed — this file never calls the Claude API

### See also

- `../agent_integration_testing/README.md` — the next layer up: testing the LOOP around these tools, not the tools themselves
- `../fixture_and_mock_management/README.md` — where the `fresh_orders()`-style fixture pattern gets promoted into a shared, reusable module
- `../../Benchmarking/trace_evaluation/README.md` — the same domain, used there to score a live agent's trace instead of unit-testing its tools
