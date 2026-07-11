# basics

The starting point: a single request/response call to the Claude API, with no conversation memory and no tools. Every other template in this repo builds on this same call shape.

## basic.py

Prompts you for a description of Python code you want, sends it to Claude, and prints the generated code.

### Concepts covered

- **Authentication** — `anthropic.Anthropic()` reads `ANTHROPIC_API_KEY` from the environment automatically; you never hardcode a key in source.
- **Core API settings** — `MODEL`, `MAX_TOKENS`, `EFFORT`, and `SYSTEM_PROMPT`, and what each one controls.
- **The `ask_claude` pattern** — build a request with a model, token limit, system prompt, and a `messages` list; read the reply back out of `response.content` (a list of content blocks, not a plain string).

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 basics/basic.py
```

You'll be prompted:

```
What Python code would you like me to generate?
>
```

Type a description (e.g. `a function that reverses a string`) and press Enter. Claude's generated code prints to the terminal.

### Configuration

Edit the constants at the top of `basic.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape how Claude generates code
