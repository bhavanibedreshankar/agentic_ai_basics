# prompt_chaining

Decomposing one complex task into a sequence of smaller, focused LLM calls, where each call's output becomes the next call's input.

## basic_prompt_chaining.py

A blog post generator that breaks the task into a chain of three focused LLM calls — outline, draft, edit — with a programmatic validation gate between the outline and the draft. Type `exit` at the topic prompt to quit.

### Concepts covered

- **Decomposing a task into steps** — `generate_outline`, `write_draft`, and `edit_draft` are each a separate, stateless call with its own narrow system prompt, instead of one call trying to do everything at once.
- **Chaining outputs into inputs** — `write_draft` builds its prompt directly from `generate_outline`'s output; `edit_draft` consumes `write_draft`'s output. That data flow *is* the chain.
- **A programmatic gate** — `validate_outline` is plain Python (no LLM call) that checks the outline has enough sections before continuing, so the chain fails fast instead of wasting a draft call on a broken outline.
- **Fixed control flow, model-generated content** — `run_chain` calls the three steps in a hardcoded sequence decided by your code, unlike `../Tools_and_Actions/tool_use/basic_agentic_tools.py`'s agentic loop, where the *model* decides what happens next and how many steps it takes. This makes it a "workflow", not an autonomous agent.
- **Narrow, single-purpose context per step** — unlike the chat templates, no shared conversation history accumulates across steps; each call only sees the specific input it needs for its one job.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 prompt_chaining/basic_prompt_chaining.py
```

Try a topic:

```
Topic: why sourdough bread takes so long to rise

[1/3] Generating outline...
- Introduction to sourdough fermentation
- The role of wild yeast and bacteria
- Temperature and its effect on rise time
- Tips for speeding up (or slowing down) fermentation
[gate] Outline has 4 sections — looks good.

[2/3] Writing draft from outline...
...

[3/3] Editing draft...

=== Final post ===
...
```

### Configuration

Edit the constants at the top of `basic_prompt_chaining.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length per step
- `EFFORT` — thinking/response depth (default: `medium`)
- `OUTLINE_SYSTEM_PROMPT` / `DRAFT_SYSTEM_PROMPT` / `EDIT_SYSTEM_PROMPT` — the focused instructions for each step
- `validate_outline` — the gate logic; edit the section-count threshold or add your own checks

### See also

- `../Tools_and_Actions/tool_use/basic_agentic_tools.py` — the contrasting pattern where the model, not your code, decides the sequence of steps
- `../Task_and_State_Management/context_management/summarization.py` — another template that chains a focused, single-purpose LLM call onto a main flow
