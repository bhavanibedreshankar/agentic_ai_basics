# agentic_loop

How a multi-turn "chat" is actually built on top of a stateless API — the loop that keeps a conversation going.

## basic_agentic_loop.py

A continuous chat agent. Keeps asking you questions and answering on any topic until you type `exit`. Maintains full conversation history across turns.

### Concepts covered

- **The API is stateless** — Claude has no memory between calls. Every request is independent, so *your* code must track everything said so far and resend the full history each time.
- **The `messages` list as the conversation** — each entry is one turn (`role` + `content`); it starts empty and grows by two entries (user, then assistant) every loop iteration.
- **The agentic loop pattern** — `while True`: get input, call the model with accumulated state, act on the result, repeat until an exit condition. This is the shape behind every conversational agent, tool-using or not.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 agentic_loop/basic_agentic_loop.py
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

- `../basics/basic.py` — the single-call building block this loop is built on
