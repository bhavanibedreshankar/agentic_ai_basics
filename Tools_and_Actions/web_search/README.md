# web_search

Web Search Tool — lets the agent retrieve live information from the internet, past its training cutoff, with no search backend for you to build.

## web_search.py

A research assistant that searches the real, live internet (via the native `web_search_20260209` server tool) for time-sensitive questions. Type `exit` to end the conversation.

### Concepts covered

- **Real search, not a local knowledge base** — contrast with every other "search" tool in this repo (`../../Task_and_State_Management/context_management/retrieval.py`, `../../RAG_and_Knowledge/embedding/`, `../../Memory/external_memory/`): those all search a small, fixed set of documents you defined ahead of time. This tool searches the actual current internet — no knowledge base to write, no embedding function to implement.
- **Server-side, same shape as `../code_interpreter/`** — declare the tool, Claude issues queries and reads results on Anthropic's infrastructure, and the queries/results just show up as extra content blocks. No `execute_tool()` here either.
- **`max_uses`** — caps how many searches can run in a single turn, a real cost/latency control since each search is billed.
- **Reading citations** — `print_response_content` pulls citation URLs off the response's text blocks and prints them as a separate "Sources" section, so it's clear where each claim actually came from, not just the prose answer.
- **The error shape** — a failed search (e.g. `max_uses_exceeded`) still returns HTTP 200 with an error object instead of a list of results — `print_response_content` checks which shape it got rather than assuming success.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Tools_and_Actions/web_search/web_search.py
```

Try:

```
You: What's the latest stable version of Python?

  [searching] latest stable Python version
  [found 4 results]

Claude: The latest stable version of Python is 3.13...

  Sources:
    - Python Downloads: https://www.python.org/downloads/
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `TOOLS`'s `max_uses` — how many searches Claude can run per turn (default: `3`); `allowed_domains`/`blocked_domains` can further scope results

### See also

- `../code_interpreter/README.md` — the other native server-side tool, same "no execute_tool()" shape
- `../../RAG_and_Knowledge/README.md` — building your own search over a knowledge base you control, for when live web results aren't what you want
