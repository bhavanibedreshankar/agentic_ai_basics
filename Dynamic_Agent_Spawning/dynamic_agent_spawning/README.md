# dynamic_agent_spawning

A meta-agent capable of creating a new sub-agent at runtime — deciding its role, persona, and system prompt on the fly — and assigning it a task, rather than delegating among a fixed roster of specialists written into the code in advance.

## dynamic_agent_spawning.py

A meta-agent with **no** built-in specialists — only one tool, `spawn_subagent`, that it can call with a role, a persona, and a task of its own choosing. Give it a request that spans expertise it couldn't have been pre-configured for and watch it invent however many specialists it actually needs. Type `exit` to quit.

### Concepts covered

- **`spawn_subagent(role, persona, task)`** — the sub-agent's entire identity (its system prompt) is assembled from strings the *parent model* chose in its own tool call, not selected from a fixed set written into this file. The same function becomes a tax accountant, a health inspector, or a marine biologist purely based on runtime `tool_input`.
- **One generic tool, not N named ones** — contrast with `../../Multi_Agent_Systems/orchestrator/orchestrator.py`, whose `delegate_to_researcher` / `delegate_to_writer` / `delegate_to_editor` are three separate, hardcoded tools with hardcoded system prompts. Here there is exactly one `spawn_subagent` tool; the number and kind of specialists that get created is unbounded and decided per-request.
- **No persistent identity** — contrast with `../../Multi_Agent_Systems/worker_agent/worker_agent.py`'s `WorkerAgent` instances, which are constructed once, ahead of time, with their own fixed tools. A sub-agent here is built and discarded inside a single `spawn_subagent` call; asking for "the same role" again later constructs an equivalent-but-new instance, not a resumed one.
- **`dispatch(name, tool_input, spawn_count)`** — enforces `MAX_SUBAGENTS_PER_TURN` as a *code* decision, never left to the model to self-limit. Kept separate from `execute_tool` specifically so the cap is unit-testable without hitting the API.
- **Fan-out in one turn** — `run_turn` walks every `tool_use` block in a single API response (a model can request several spawns at once) and answers all of them in one combined `tool_result` message before continuing the loop, the same multi-block handling used in `../../Multi_Agent_Systems/orchestrator/orchestrator.py`'s `run_turn`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Dynamic_Agent_Spawning/dynamic_agent_spawning/dynamic_agent_spawning.py
```

Try:

```
You: I'm signing a lease to run a bakery out of a rented commercial kitchen — what should I watch out for?

  [spawn request: Commercial Lease Attorney] task: Review key lease terms a small food business should scrutinize...
  [Commercial Lease Attorney -> ] Watch for exclusive-use clauses, maintenance responsibility...

  [spawn request: Health Code Inspector] task: What health code requirements apply to a rented commercial kitchen...
  [Health Code Inspector -> ] You'll need a shared-kitchen permit, documented cleaning schedules...

  [spawn request: Small Business Tax Advisor] task: What tax considerations apply to renting a commercial kitchen...
  [Small Business Tax Advisor -> ] Rent is generally deductible as a business expense...

Meta-agent: Here's what to watch for across three areas: ...
```

Ask something in a completely unrelated domain next (e.g. "help me plan a birthday trip that involves diving and a tight budget") and notice the specialists it spawns are entirely different — nothing here was written for bakeries or lawyers specifically.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `SUBAGENT_MAX_TOKENS` — token budget for each spawned sub-agent's single-shot answer
- `MAX_SUBAGENTS_PER_TURN` — how many sub-agents the meta-agent may spawn in one turn before `dispatch()` starts rejecting further spawns (default: `4`)
- `META_AGENT_SYSTEM_PROMPT` — the instruction that tells the parent it has no built-in specialists and must invent them

### See also

- `../../Multi_Agent_Systems/orchestrator/README.md` — the fixed-roster version of delegation this template deliberately removes the "fixed" part of
- `../../Multi_Agent_Systems/worker_agent/README.md` — a sub-agent with its own internal tool-calling loop, pre-instantiated rather than assembled at call time
- `../../Self_Evolving_Agents/self_evolving_agents/README.md` — another template where a cap on a model-controlled quantity (rules learned, sub-agents spawned) is enforced entirely in code
