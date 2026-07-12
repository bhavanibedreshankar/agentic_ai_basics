# self_evolving_agents

Self-evolving agents are AI systems that autonomously update their own capabilities — such as prompts, tools, memory, and reasoning workflows — through continuous, closed-loop feedback. This template demonstrates that loop on the system prompt: the piece of the agent that shapes every future answer.

## self_evolving_agents.py

A coding-help assistant. Ask a question, rate the answer thumbs-down with a reason if it's missing something, and the agent distills that feedback into a rule it appends to its own system prompt — permanently, in `evolved_rules.json`. Ask a similar question again (even in a brand-new run of the script) and the rule is already in effect. Type `exit` to quit.

### Concepts covered

- **`build_system_prompt(rules)`** — the agent's live instructions are its fixed base prompt plus everything it has learned so far. Called fresh before every answer, so a rule added a minute ago (or in yesterday's run) is already active — contrast with `../../Core_Architecture/system_prompt/system_prompt.py`'s prompt, which is authored once and never changes itself.
- **`propose_rule(question, response_text, feedback)`** — turns one piece of negative feedback into a single, *general* instruction (not a fix for that one question), using the same `{schema}` + `json.loads` structured-output pattern as `../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py`.
- **`evolve(...)`** — the closed loop itself: propose a rule, dedupe it against what's already learned, persist it to disk, and it's live for the next call. Kept separate from `propose_rule` specifically so the cap/dedupe policy is unit-testable without an API call.
- **`MAX_RULES`** — a hard cap on self-modification. Without one, a noisy or adversarial feedback stream could let the agent rewrite its own instructions without bound — same spirit as the caps in `../../Safety_and_Control/guardrails.py`, applied to the agent's own prompt instead of its actions.
- **Persistence across processes, not just across turns** — contrast with `../../Planning_and_Reasoning/self_reflection/self_reflection.py`, whose critique-and-revise loop improves one output and forgets everything once `main()` returns.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py
```

Try:

```
Question: How do I read a file in Python?

You can read a file using open():

    with open("data.txt") as f:
        contents = f.read()

Rate this answer (up/down, or press enter to skip): down
What was wrong with it? didn't say what to do if the file doesn't exist
[agent evolved] added rule: Always mention how to handle the case where the target file does not exist.

Question: How do I write to a file in Python?

You can write to a file using open() in write mode. If the destination
directory doesn't exist, this will raise a FileNotFoundError, so make
sure the path is valid first:

    with open("data.txt", "w") as f:
        f.write("hello")
```

Notice the second answer already accounts for a missing-path failure mode — nobody told it to this time.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `BASE_SYSTEM_PROMPT` — the fixed instructions every version of the agent starts from
- `MAX_RULES` — how many learned rules the agent can accumulate before it stops adding more (default: `8`)
- `evolved_rules.json` — the persisted rule set; delete it to reset the agent to its base behavior

### See also

- `../../Planning_and_Reasoning/self_reflection/README.md` — the ephemeral, single-output version of "critique and improve," with nothing persisted
- `../../Memory/episodic_memory/README.md` — persists a log of past interactions for the agent to *recall*, rather than distilling them into instructions that apply automatically
- `../../Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — the structured-output scoring pattern reused here for turning feedback into a rule
- `../../Safety_and_Control/guardrails/README.md` — bounding what an agent can do; here applied to bounding what an agent can do to *itself*
