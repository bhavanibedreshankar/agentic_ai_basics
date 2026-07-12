# system_prompt

System Prompt — the fixed instruction set that defines an agent's role, constraints, and available tools, sent with every request separate from the user's message.

## system_prompt.py

Sends the identical user question through the same call shape as `../basics/basic.py`'s `ask_claude`, three times, under three different system prompts — a terse expert, an ELI5 tutor, and a strict JSON-only API. Everything about the request is the same except the system prompt.

### Concepts covered

- **The system prompt alone explains the whole difference** — `run_with_prompt` is one fixed function; `compare_prompts` calls it three times with the same `user_message` and only `system_prompt` varying. Any difference in tone, length, format, or what counts as an in-scope answer comes from that one parameter.
- **Role vs. constraint vs. format, in one instruction set** — `terse_expert` sets a role (senior engineer) and a hard constraint (two sentences max); `eli5_tutor` sets a different role and a different constraint (no jargon, use an analogy); `strict_json_api` discards "assistant" framing entirely and constrains the output to a single JSON shape with no prose. All three are just strings passed to the same `system=` parameter.
- **Contrast with `../basics/basic.py`** — that file picks one `SYSTEM_PROMPT` and never varies it, so its effect is easy to take for granted. This file makes the same mechanism visible by holding the user's message constant and only changing the system prompt.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Core_Architecture/system_prompt/system_prompt.py
```

Try (or press enter for the built-in default):

```
Question to send under every system prompt (blank for a default demo):
> Why does my Python code throw a KeyError when I access a dictionary?

Same question, 3 different system prompts:
  Why does my Python code throw a KeyError when I access a dictionary?

--- terse_expert ---
A KeyError happens when you access a dict key that doesn't exist. Use .get() or check with 'in' first.

--- eli5_tutor ---
Think of a dictionary like a row of labeled lockers -- each key is a label. If you ask for a locker
label that isn't there, Python can't find it and throws a KeyError to let you know!

--- strict_json_api ---
{"summary": "KeyError on dict access", "likely_cause": "requested key not present in the dictionary"}
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../basics/README.md`
- `SYSTEM_PROMPTS` — the named dict of system prompts compared; add or edit entries to try your own role/constraint combinations

### See also

- `../basics/README.md` — the fixed-`SYSTEM_PROMPT` version of the same call shape this file varies
- `../llm_backbone/README.md` — the same "hold everything constant except one parameter" comparison technique, applied to the model instead of the system prompt
- `../tool_use/README.md` — where a system prompt's description of available tools shapes behavior directly, rather than just tone/format
