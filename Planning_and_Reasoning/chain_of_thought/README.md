# chain_of_thought

Chain-of-Thought (CoT) — prompting the model to reason step by step before producing a final answer.

## chain_of_thought.py

Shows the same question answered three ways — direct, prompted CoT, and native extended thinking — side by side, so the difference is visible. Type `exit` to quit.

### Concepts covered

- **`answer_direct`** — no reasoning requested. Fast, but on multi-step problems this is where a model is most likely to jump to a plausible-sounding wrong answer.
- **`answer_with_cot_prompting`** — the classic technique (Wei et al., 2022): explicitly instructing the model to "think step by step" and show its work as visible response text before stating a final answer. Works on any model, no special API parameter required.
- **`answer_with_native_thinking`** — Claude's built-in extended thinking (`thinking: {"type": "adaptive"}`). The model reasons in a separate `thinking` content block, distinct from its final answer — the modern, API-native equivalent of prompted CoT.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Planning_and_Reasoning/chain_of_thought/chain_of_thought.py
```

Try a multi-step problem:

```
Question: A store has 15 apples. They sell 40% of them in the morning and 6 more in the afternoon. How many are left?

--- 1. Direct (no reasoning) ---
...
--- 2. Prompted chain-of-thought ---
...
--- 3. Native extended thinking ---
[thinking]
...
[answer]
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- Each `answer_*` function's `system` prompt controls that variant's behavior directly

### See also

- `../react/README.md` — interleaving this same kind of reasoning with tool calls, one action at a time
- `../tree_of_thought/README.md` — running several reasoning attempts in parallel instead of just one
