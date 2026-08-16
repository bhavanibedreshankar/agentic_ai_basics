# agent_integration_testing

Testing the full multi-turn, multi-tool-call agentic loop end-to-end with a stubbed/mocked LLM client, so orchestration logic (loop termination, message history, parallel tool calls, retries) is verified deterministically and without live API cost.

## agent_integration_testing.py

A pytest-style test module exercising `run_agent` — a tool-calling loop with the same shape as `../../Core_Architecture/tool_use/basic_agentic_tools.py`'s `run_turn`, except the LLM client is a constructor parameter instead of a module-level singleton, so tests can hand it a scripted `FakeClient` instead of hitting the real API. Run with `pytest agent_integration_testing.py` or directly with `python3 agent_integration_testing.py`.

### Concepts covered

- **`run_agent(client, task)`** — the loop under test; unlike `../tool_unit_testing/`, this tests the ORCHESTRATION around tool calls (call count, message ordering, stop conditions), not what any one tool returns.
- **`FakeClient` / `text_response` / `tool_use_response`** — hand-built response builders and a queue-based stub for `anthropic.Anthropic()`; `client.call_count` lets tests assert on exactly how many API calls happened.
- **`test_parallel_tool_calls_all_execute_and_match_ids`** — proves multiple `tool_use` blocks in one response all get executed and their `tool_use_id`s round-trip correctly, including one that resolves to an error without crashing the loop.
- **`test_transient_error_is_retried_and_succeeds` / `test_retries_exhausted_raises`** — `_create_with_retry`'s retry-at-the-orchestration-layer behavior, proven with a fake client that raises on its first call and either recovers or doesn't.
- **`_run_tests_manually()`** — same dependency-free runner as `../tool_unit_testing/tool_unit_testing.py`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 Agent_Testing/agent_integration_testing/agent_integration_testing.py
# or
pytest Agent_Testing/agent_integration_testing/agent_integration_testing.py
```

Try:

```
  [PASS] test_parallel_tool_calls_all_execute_and_match_ids
  [PASS] test_retries_exhausted_raises
  [PASS] test_single_tool_call_then_final_answer
  [PASS] test_single_turn_no_tools_needed
  [PASS] test_transient_error_is_retried_and_succeeds

5 passed, 0 failed
```

### Configuration

- `MAX_RETRIES` — how many times `_create_with_retry` retries a failed `client.messages.create` call before raising
- No `ANTHROPIC_API_KEY` needed — `FakeClient` never touches the network

### See also

- `../tool_unit_testing/README.md` — the layer below: testing tool functions in isolation, which this file assumes already work
- `../fixture_and_mock_management/README.md` — extracts `FakeClient`/`text_response`/`tool_use_response` into a shared, importable toolbox instead of this file's own hand-rolled copies
- `../../Benchmarking/trace_evaluation/README.md` — the same kind of loop run for real, then scored for quality rather than tested for orchestration correctness
