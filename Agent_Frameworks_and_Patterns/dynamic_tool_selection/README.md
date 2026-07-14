# dynamic_tool_selection

An agent that draws from a large pool of tool "libraries" but only loads the tools relevant to the current request into its context, instead of sending every tool's full definition on every call.

## dynamic_tool_selection.py

A personal-productivity assistant with 10 tools spanning 6 unrelated libraries — weather, finance, calendar, email, GitHub, and notes. Every tool is marked `defer_loading: True` except the search tool itself, so no matter what you ask, only the 1-3 actually-relevant tools ever enter Claude's context. Type `exit` to quit.

### Concepts covered

- **`tool_search_tool_bm25_20251119`** — Anthropic's real, native tool-search server tool (no beta header, generally available), declared the same way as `../../Tools_and_Actions/web_search/web_search.py`'s `web_search`. BM25 takes natural-language search queries; the sibling `tool_search_tool_regex_20251119` variant takes Python regex patterns instead.
- **`defer_loading: True`** — controls what enters Claude's *context*, not what you *transmit*. Every one of the 10 tools in `DEFERRED_TOOLS` is sent in full on every request (the API needs the complete schema server-side to search over), but stays out of the model's attention until a search surfaces it.
- **One response, one round trip** — per Anthropic's docs, a single API response can contain the search call, the search result, *and* the discovered tool's call, all before returning to you. `run_turn` handles this by iterating every block type in one pass: `server_tool_use` (the search call — never gets a `tool_result`), `tool_search_tool_result` (search results — just appended to history), and `tool_use` (a discovered tool — executed exactly like standard tool use).
- **`execute_tool(name, tool_input)`** — dispatch for the 10 *discovered* client-side tools only, same registry-style single dict lookup as `../tool_registry/basic_tool_registry.py`'s `execute_tool`. The search tool itself has no handler here; it runs entirely on Anthropic's infrastructure.
- **Scales past what a static catalog can handle** — contrast with `../tool_registry/basic_tool_registry.py`, whose `build_tool_catalog()` sends its whole registry into context every time. That file's own closing note says this doesn't scale past a handful of tools and names tool search as the fix — this template is that fix in practice.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Agent_Frameworks_and_Patterns/dynamic_tool_selection/dynamic_tool_selection.py
```

Try:

```
You: What's the weather in Austin, and convert 100 miles to km.

  [tool search] 'weather'
  [discovered] ['weather_get_current']
  [calling] weather_get_current({'location': 'Austin'})
  [result] Austin: 91°F, sunny

  [tool search] 'unit conversion'
  [discovered] ['utils_convert_units']
  [calling] utils_convert_units({'value': 100, 'from_unit': 'miles', 'to_unit': 'km'})
  [result] 100 miles = 160.93 km

Assistant: It's 91°F and sunny in Austin. 100 miles is about 160.93 km.
```

Notice only 2 of the 10 tools were ever discovered or loaded — ask about GitHub issues or notes instead and you'll see a completely different pair surface.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `TOOL_SEARCH_TOOL` — swap `tool_search_tool_bm25_20251119` for `tool_search_tool_regex_20251119` to have Claude search with regex patterns instead of natural language
- `DEFERRED_TOOLS` — the 10-tool catalog; Anthropic recommends keeping your 3-5 most-used tools non-deferred in production (this demo defers all of them so every discovery is visible)

### See also

- `../tool_registry/README.md` — the same registry idea at a scale where sending the whole catalog every time is still fine
- `../../Tools_and_Actions/web_search/README.md` — another native Anthropic server tool declared for real, with the same "server-side call gets no `execute_tool`" pattern
- `../../Dynamic_Agent_Spawning/dynamic_agent_spawning/README.md` — selecting among existing tools at runtime vs. inventing a new agent persona at runtime
