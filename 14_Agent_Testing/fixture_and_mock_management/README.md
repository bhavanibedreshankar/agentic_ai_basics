# fixture_and_mock_management

Building reusable, shared fixtures for canned LLM responses and tool side effects (fake databases, sandboxed filesystems) so the rest of the suite is fast, deterministic, and repeatable rather than hitting real external systems or paying for live API calls.

## fixture_and_mock_management.py

A pytest-style test module that extracts the ad hoc fake-response and fake-state helpers `../tool_unit_testing/tool_unit_testing.py` and `../agent_integration_testing/agent_integration_testing.py` each hand-rolled independently into one shared, reusable toolbox — then proves the payoff by composing them into a full scenario in a handful of lines. Run with `pytest fixture_and_mock_management.py` or directly with `python3 fixture_and_mock_management.py`.

### Concepts covered

- **`text_block` / `tool_use_block` / `response`** — the atomic builders every fake Claude response is assembled from, matching `anthropic.types.Message`'s shape (`.content`, `.stop_reason`, per-block `.type`/`.text`/`.name`/`.input`/`.id`) in exactly one place instead of one place per test file.
- **`FakeClient`** — the shared, reusable version of `../agent_integration_testing/agent_integration_testing.py`'s queue-based stub, now also recording every call's kwargs in `self.calls` so tests can assert on what was SENT, not just what came back.
- **`fresh_orders_db(overrides=None)`** — the generalized version of `../tool_unit_testing/tool_unit_testing.py`'s `fresh_orders()`, now accepting overrides so a test can start from a specific scenario without hand-writing the whole fixture.
- **`test_composing_fixtures_for_integration_scenario`** — the payoff: the same multi-step scenario `../agent_integration_testing/agent_integration_testing.py` needed ~20 lines for, built here in about five, from the shared toolbox.
- **`_run_tests_manually()`** — same dependency-free runner as `../tool_unit_testing/tool_unit_testing.py`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 14_Agent_Testing/fixture_and_mock_management/fixture_and_mock_management.py
# or
pytest 14_Agent_Testing/fixture_and_mock_management/fixture_and_mock_management.py
```

Try:

```
  [PASS] test_composing_fixtures_for_integration_scenario
  [PASS] test_fake_client_records_call_arguments
  [PASS] test_fresh_orders_db_supports_overrides
  ...

8 passed, 0 failed
```

### Configuration

- No configuration constants — this file is pure test infrastructure
- No `ANTHROPIC_API_KEY` needed — `FakeClient` never touches the network

### See also

- `../tool_unit_testing/README.md` and `../agent_integration_testing/README.md` — the two files whose ad hoc fixtures this template generalizes and consolidates
- `../test_suite_pyramid/README.md` — as a suite scales past a handful of files, this is the module every fast-tier test file would import from instead of copy-pasting builders
