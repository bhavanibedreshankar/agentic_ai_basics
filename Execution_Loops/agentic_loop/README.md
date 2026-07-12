# agentic_loop

Agentic Loop — the core observe → think → act → observe cycle an agent runs until the task is done or halted, in its simplest possible form: a multi-turn chat built on top of a stateless API.

## basic_agentic_loop.py

A continuous chat agent. Keeps asking you questions and answering on any topic until you type `exit`. Maintains full conversation history across turns.

### Concepts covered

- **The API is stateless** — Claude has no memory between calls. Every request is independent, so *your* code must track everything said so far and resend the full history each time.
- **The `messages` list as the conversation** — each entry is one turn (`role` + `content`); it starts empty and grows by two entries (user, then assistant) every loop iteration.
- **The agentic loop pattern** — `while True`: get input, call the model with accumulated state, act on the result, repeat until an exit condition. This is the shape behind every conversational agent, tool-using or not.

### Mapping the loop onto observe → think → act → observe

This template's loop is the cycle at its most stripped-down — worth naming explicitly, since every other template in `Execution_Loops/` and most of the rest of the repo is a variation on it:

| Phase | In this template |
|---|---|
| **Observe** | `input("You: ")` — read what the user just said |
| **Think** | the API call itself — Claude reasons over the accumulated `messages` history |
| **Act** | `print(f"\nClaude: {reply}\n")` — the only "action" here is responding in text; there's no tool call, so nothing actually changes in the world |
| **Observe** (again) | back to `input("You: ")` for the next cycle |

The honest limitation: because this loop never calls a tool, "Act" never touches anything external, and there's nothing for a subsequent "Observe" to notice *besides* the user's next message — the cycle only closes through the human, not through the environment. For the version where Act means calling a tool and Observe means reading back that tool's real result — the fuller form of the cycle — see `../../Tools_and_Actions/tool_use/basic_agentic_tools.py`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Execution_Loops/agentic_loop/basic_agentic_loop.py
```

You'll see:

```
Chat with Claude. Type 'exit' to end the conversation.

You:
```

Type a message and press Enter to get a response. Keep chatting, or type `exit` to end the conversation.

### Configuration

Edit the constants at the top of `basic_agentic_loop.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant

### See also

- `../../basics/basic.py` — the single-call building block this loop is built on
