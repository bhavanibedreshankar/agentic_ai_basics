# trace_evaluation

Scoring an agent's entire execution trace — every tool call, tool result, and intermediate step — instead of only its final answer.

## trace_evaluation.py

Runs a support agent (order lookup / refund-eligibility / issue-refund tools) live once, then evaluates its captured trace two ways: deterministic rule-based checks and an LLM judge that reads the whole step sequence. A second demo runs the same two evaluators against a hand-built "bad" trace (refund issued with no eligibility check, and a final answer that states a fact no tool result ever supported) to show both catching a real defect on demand. Type `exit` to quit after the demos run, or supply your own task.

### Concepts covered

- **`TraceStep` / `run_agent(task)`** — captures every tool call, tool result, and final answer from a real tool-calling loop (same inner-loop shape as `../../01_Core_Architecture/tool_use/basic_agentic_tools.py`'s `run_turn`) into a plain ordered list, independent of the Claude SDK's own message format.
- **`check_trace_rules(trace)`** — deterministic, no LLM call: flags redundant tool calls, too many steps (`MAX_TOOL_CALLS`), and a domain precondition violation (`issue_refund` called without a prior `check_refund_eligibility` for the same order). Catches structural bugs a judge reading only the final answer would never see.
- **`judge_trace(task, trace)`** — the same structured-output judge shape as `../llm_judge_benchmarking/llm_judge_benchmarking.py`'s `evaluate_output` (score + code-side `passed` against `PASS_THRESHOLD`), but scored against the full `format_trace(trace)` text instead of just a final output — so it can score groundedness (does the final answer only state facts a tool actually returned?) and error recovery, not just tone or correctness of the ending.
- **`evaluate_trace(task, trace)`** — combines both: rules run first (instant, free), the judge runs second (semantic, but non-deterministic) — the same "layer a free deterministic check before an expensive model call" idea used throughout the repo's tool dispatchers.
- **`BAD_TRACE`** — a hand-built defective trace (skipped precondition + ungrounded claim), used so the demo can reliably show a caught violation without depending on the live model happening to misbehave.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 13_Benchmarking/trace_evaluation/trace_evaluation.py
```

Try:

```
=== Demo 2: a hand-built BAD trace (skipped eligibility check, ungrounded claim) ===

--- rule-based checks ---
  [VIOLATION] precondition_skipped: issue_refund called for order 1002 with no prior check_refund_eligibility call

--- LLM-as-judge over the trace ---
  groundedness=2/10  task_adherence=6/10  error_recovery=1/10
  average=3.0/10  passed=False
  feedback: The final answer claims the order was "confirmed eligible," but no
  check_refund_eligibility call ever occurred, and the order is actually
  outside the refund window.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../01_Core_Architecture/basics/README.md`
- `PASS_THRESHOLD` — average judge score cutoff (default: `7` out of 10)
- `MAX_TOOL_CALLS` — rule-based efficiency ceiling on tool calls per task (default: `4`)
- `REFUND_WINDOW_DAYS` / `ORDERS` — the mock domain data driving eligibility

### See also

- `../llm_judge_benchmarking/README.md` — the single-output judge this template extends to score a whole trace instead
- `../task_accuracy_eval/README.md` — the deterministic final-answer counterpart this template's rule checks are structurally closer to, just applied to trace shape instead of an answer string
- `../../17_LangChain/callbacks_and_tracing/README.md` and `../../10_Safety_and_Control/audit_trail/README.md` — capturing a trace as an agent runs, the step this template assumes has already happened
- `../../08_Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — the reusable single-call scoring primitive both `check_trace_rules`'s pass/fail philosophy and `judge_trace` borrow from
