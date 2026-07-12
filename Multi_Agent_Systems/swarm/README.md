# swarm

Swarm — a loosely coordinated collection of agents working in parallel, often without a central orchestrator.

## swarm.py

An investment proposal swarm: three independent analysts (financial, risk, market) evaluate the same proposal simultaneously with no awareness of each other, then a synthesis step combines all three perspectives. Type `exit` to quit.

### Concepts covered

- **No coordinator, genuinely parallel** — contrast with `../orchestrator/`: there, one agent deliberately decides who does what, in sequence. Here, every analyst gets the same input at the same time via `concurrent.futures.ThreadPoolExecutor`, and none of them knows the others exist.
- **Merge everything, discard nothing** — contrast with `../../Planning_and_Reasoning/tree_of_thought/tree_of_thought.py`: ToT runs several attempts at the *same* problem and picks the single best one. A swarm's members answer *different* questions about the same proposal (financial vs. risk vs. market), so `synthesize` combines all of them — there's no "best" one to pick.
- **The first genuinely concurrent template in this repo** — every other multi-call template calls Claude sequentially. `run_swarm` fires all three analysts' API calls at once; verified in testing that three simulated 0.5-second calls complete in ~0.5s total, not ~1.5s, proving the concurrency is real rather than just structured to look parallel.
- **`as_completed`, not submission order** — results are collected in whichever order they actually finish, not the order the analysts were listed in `ANALYSTS`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Multi_Agent_Systems/swarm/swarm.py
```

Try:

```
Proposal: We're proposing to launch a subscription box for artisanal coffee.
  [market analyst done in 2.1s]
  [risk analyst done in 2.3s]
  [financial analyst done in 2.4s]
  [swarm total wall-clock time: 2.4s — compare to the sum of individual times above]

--- Financial analyst ---
...

=== Synthesized recommendation ===
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `ANALYSTS` — the swarm members and their system prompts; add another perspective by adding an entry here

### See also

- `../orchestrator/README.md` — the deliberate, sequential alternative to this template's uncoordinated parallelism
- `../../Planning_and_Reasoning/tree_of_thought/README.md` — parallel exploration that picks one winner, instead of merging every result
