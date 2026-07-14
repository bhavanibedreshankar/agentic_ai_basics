# procedural_memory

An agent's own instructions improving over time from feedback, persisted so future runs start from the upgraded prompt.

## procedural_memory.py

A support assistant whose system prompt evolves in place, persisted to `evolved_prompt.json`. Give it feedback after an answer and the prompt updates for the very next turn — in this run and in a fresh process. Type `exit` to end the session.

### Concepts covered

- **`create_prompt_optimizer(llm, kind="prompt_memory")`** — reads the current prompt plus an `AnnotatedTrajectory` (conversation + feedback) and returns a rewritten prompt, deciding on its own how to incorporate the lesson.
- **`AnnotatedTrajectory(messages=..., feedback=...)`** — pairs a conversation with feedback on it, the unit of input the optimizer reasons over.
- **`load_prompt` / `save_prompt`** — persistence to `evolved_prompt.json` (gitignored), read at the START of every turn so an update from one turn is live for the next.
- Contrast with [`../../Self_Evolving_Agents/self_evolving_agents/README.md`](../../Self_Evolving_Agents/self_evolving_agents/README.md), which distills feedback into one explicit, inspectable RULE and splices it into the prompt by hand — here the rewrite itself is the optimizer's judgment call, not a discrete appended rule.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langchain langgraph langmem
export ANTHROPIC_API_KEY=your-key-here
python3 LangMem/procedural_memory/procedural_memory.py
```

Try:

```
Current prompt: 'You are a helpful support assistant.'

You: What's your refund policy?
Claude: I don't have specific details on that.

Feedback on that reply (blank to skip): Should have said refunds are available within 30 days.

[prompt updated]
  before: 'You are a helpful support assistant.'
  after:  'You are a helpful support assistant. Always mention the 30-day refund policy when relevant.'
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `DEFAULT_PROMPT` — the starting prompt before any feedback is given
- `kind="prompt_memory"` in `build_optimizer` — LangMem's simplest optimizer; `"gradient"` and `"metaprompt"` run a multi-step reflect-then-revise process for harder cases

### See also

- [`../../Self_Evolving_Agents/self_evolving_agents/README.md`](../../Self_Evolving_Agents/self_evolving_agents/README.md) — the hand-built version: explicit rule distillation and splicing
