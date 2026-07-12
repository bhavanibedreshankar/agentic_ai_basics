# react

ReAct (Reason + Act) — a prompting pattern where the model alternates between explicit reasoning steps and tool-calling actions, narrating a Thought before every Action.

## react_agent.py

A research agent with two tools (`lookup_fact`, `calculate`) answering questions that need multiple lookups and a calculation to solve — forcing the Thought → Action → Observation cycle to repeat more than once per question. Type `exit` to end the conversation.

### Concepts covered

- **Explicit, labeled reasoning** — the `SYSTEM_PROMPT` instructs Claude to narrate a `Thought:` before every tool call, making the reasoning behind each action visible and auditable, not just the action itself.
- **Contrast with `../../Core_Architecture/tool_use/basic_agentic_tools.py`** — structurally the same loop (call tools, feed results back), but ReAct specifically makes the reasoning between actions part of the visible transcript. That's what turns plain tool use into ReAct.
- **Contrast with native extended thinking** (`../chain_of_thought/`) — a related but distinct mechanism. ReAct's reasoning is prompted, visible response text; native thinking is a separate `thinking` block the API produces on its own.
- **Fuzzy fact lookup** — `lookup_fact` matches by word overlap rather than requiring an exact key, so differently-worded lookups (e.g. "france population" vs. "population of france") still resolve.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Planning_and_Reasoning/react/react_agent.py
```

Try:

```
You: What's the combined population of France and Germany, and how much would 3 widgets and 2 gadgets cost?

Thought: I need to look up the population of France first.
Action: lookup_fact({'topic': 'population of france'})
Observation: France has a population of approximately 68 million.
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `SYSTEM_PROMPT` — the instruction that establishes the Thought/Action/Observation pattern; this *is* the ReAct technique
- `FACTS` — the mock knowledge base `lookup_fact` searches

### See also

- `../../Core_Architecture/tool_use/README.md` — the underlying tool-calling loop this template adds explicit reasoning to
- `../chain_of_thought/README.md` — reasoning before a single answer, without tool calls or actions
