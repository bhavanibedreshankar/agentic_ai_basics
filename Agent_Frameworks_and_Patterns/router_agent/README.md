# router_agent

Router Agent — an agent that classifies the input and routes it to the appropriate sub-agent or tool, as a single upfront decision before any response generation.

## router_agent.py

A support-ticket router classifying incoming messages into `billing` / `technical` / `sales` / `general`, each handled by its own focused call. Type `exit` to quit.

### Concepts covered

- **Classify-then-dispatch, not tool-based transfer** — contrast with `../../Multi_Agent_Systems/agent_handoff/agent_handoff.py`: handoff routes via a `transfer_to_X` tool the *model* chooses to call mid-conversation. Here, `classify()` is a single structured-output call that runs *before* any response generation, and `route()` dispatches based on its result — the same "call 1 feeds call 2" shape as `../prompt_chaining/basic_prompt_chaining.py`, not a tool-calling loop.
- **A closed category set, never free text to parse** — `CLASSIFY_SCHEMA` constrains the classifier's output to an `enum` of exactly four categories via structured output, so there's no risk of the classifier inventing a category or phrasing one ambiguously.
- **Confidence-gated fallback** — the classifier also reports `confident`; a low-confidence classification is routed to `general` instead of the raw (possibly wrong) guess. Verified directly: a classification of `'sales'` with `confident: False` results in the `general` handler being used, not `sales`.
- **A natural fit for a cheaper classifier model** — the docstring notes that because classification is independent of the final response call, a real system could route the `classify()` call to a smaller/faster model without touching any handler.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Agent_Frameworks_and_Patterns/router_agent/router_agent.py
```

Try:

```
Message: I was charged twice this month
  [router] classified as 'billing' (confident)

[billing] I'm sorry to hear about the duplicate charge...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `CATEGORIES` / `CLASSIFY_SCHEMA` — the closed set of routes
- `HANDLER_SYSTEM_PROMPTS` — each category's dedicated persona

### See also

- `../../Multi_Agent_Systems/agent_handoff/README.md` — the contrasting tool-based, mid-conversation routing mechanism
- `../prompt_chaining/README.md` — the general "one call feeds the next" shape this template's classify-then-dispatch flow follows
