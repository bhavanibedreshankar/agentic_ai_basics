# max_iterations

Max Iterations / Turn Limit — a safety cap on how many loop cycles an agent may run before stopping, unconditionally.

## max_iterations.py

A research assistant with a deliberately "flaky" search tool, capped at `MAX_ITERATIONS = 5` loop cycles. Type `exit` to quit.

### Concepts covered

- **No condition to evaluate — just a count** — contrast with `../interrupts_breakpoints/interrupts_breakpoints.py`: that template's breakpoints only fire when a specific condition is met, and in principle could let a task run forever if none of them ever trigger. `run_turn` here has no such gap: `iteration` increments every pass and the loop stops at `max_iterations` regardless of what's happening in the task.
- **The check happens before the call it would prevent** — `if iteration > max_iterations: return ...` runs at the *top* of the loop, before that iteration's API call — the (N+1)th call genuinely never happens, not just "the loop looks like it stopped."
- **Proven with a non-converging simulated agent** — a mocked API response that always requests another tool call (simulating a bug or bad prompt that never resolves) is capped at exactly 5 real API calls, verified by counting them, not just checking the printed output.
- **Doesn't waste budget on early completion** — a second test simulates an agent that converges on iteration 2 of a 5-iteration cap, and confirms the loop stops immediately at 2 calls rather than continuing to iterate needlessly.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Execution_Loops/max_iterations/max_iterations.py
```

Try a query the mock tool won't resolve, to watch the cap trigger:

```
You: Look up the shipping policy for international orders.
  [iteration 1/5]
  [tool] flaky_search({'query': 'international shipping policy'})
  [result] No useful results found for that query. Try rephrasing.
  [iteration 2/5]
  ...
  [iteration 5/5]
  [tool] flaky_search({'query': '...'})
  [result] No useful results found for that query. Try rephrasing.

=== stopped: reached MAX_ITERATIONS (5) without a final answer ===
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `MAX_ITERATIONS` — the cap (default: `5`, deliberately low so it's reachable in a demo session — a real agent would set this much higher, sized to the task)
- `KNOWN_QUERIES` — the one query `flaky_search` will actually resolve, for testing the "completes before the cap" path

### See also

- `../interrupts_breakpoints/README.md` — the more targeted, condition-based alternative this template's unconditional cap backstops
- `../human_in_the_loop/README.md` — pausing on specific actions rather than a total iteration count
