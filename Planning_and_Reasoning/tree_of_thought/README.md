# tree_of_thought

Tree of Thought (ToT) — extending chain-of-thought to explore multiple reasoning branches in parallel and select the best path, instead of committing to a single line of reasoning.

## tree_of_thought.py

Solves a problem three different ways (via three explicit strategies), evaluates each attempt, and selects the highest-scored one. Demonstrated on the classic "bat and ball" puzzle, where a fast intuitive answer is a well-known wrong answer. Type `exit` to quit.

### Concepts covered

- **A simplified, single-level ToT** — the full technique (Yao et al., 2023) recurses branch/evaluate/prune at every reasoning step, searched many levels deep. This template does one round: generate N candidate approaches, evaluate each, pick the best — the core mechanic without the multi-level search.
- **Explicit strategy diversity instead of randomness** — `BRANCH_STRATEGIES` gives each branch a genuinely different instruction (algebra vs. logical reasoning vs. guess-and-check) rather than relying on sampling randomness for variety, which isn't configurable via `temperature` on current models anyway.
- **`evaluate_branch`** — a separate, focused call whose only job is scoring a candidate solution (0–10) against the actual problem, independent of how that candidate was generated.
- **`select_best_branch`** — orchestrates the full round and returns the winning branch, along with every branch's score for comparison.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Planning_and_Reasoning/tree_of_thought/tree_of_thought.py
```

Try:

```
Problem: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

=== Branches ===

[Branch 1] strategy: Solve this using algebra...
...
  -> score: 10/10 (correct, ball = $0.05, verified against both conditions)

=== Selected answer (highest-scored branch) ===
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `BRANCH_STRATEGIES` — the list of distinct approaches; add, remove, or reword entries to change how the branches diverge
- `EVALUATOR_SYSTEM_PROMPT` — the criteria the evaluator judges each branch against

### See also

- `../chain_of_thought/README.md` — the single-path reasoning this template runs multiple times in parallel
- `../self_reflection/README.md` — improving *one* output iteratively, instead of comparing several independent ones
