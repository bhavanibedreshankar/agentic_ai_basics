# api_connectors_mcp

API Connectors / MCP — integrating an agent with an external service: either by wrapping a REST call as a custom tool, or by connecting to a Model Context Protocol (MCP) server that already exposes one.

## api_connectors_mcp.py

A weather assistant that calls a real, live, public REST API (Open-Meteo — free, no API key required) as a custom tool. Type `exit` to end the conversation. The bottom of the file has a commented, non-executed reference for the MCP connector alternative.

### Concepts covered

- **A real external API call, standard library only** — `get_weather` uses `urllib.request` (no `requests`, no new dependency) to hit a live weather API and parse the JSON response. Same tool-definition/`execute_tool` shape as every custom tool elsewhere in this repo (`../../tool_use/`) — the only thing new is that the implementation makes a real network call instead of computing something locally.
- **Why MCP is shown, not run** — an MCP server needs a real URL (and often authentication) to connect to; hardcoding one to demonstrate against would mean depending on infrastructure this template doesn't control and can't guarantee stays available. The commented block at the bottom shows the exact request shape (`mcp_servers` + a `mcp_toolset` tool entry) for when you have a real server to point it at.
- **MCP is server-side too** — like `../code_interpreter/` and `../web_search/`, an MCP-connected tool has no `execute_tool()` on your side: the MCP server (run by you or by whoever operates the service) executes the call over the network, and Claude talks to it directly.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Tools_and_Actions/api_connectors_mcp/api_connectors_mcp.py
```

Try:

```
You: What's the weather like in Tokyo right now?
  [tool] get_weather({'city': 'Tokyo'})
  [result] Tokyo: 25.7°C, wind 1.5 km/h

Claude: It's currently 25.7°C in Tokyo with a light wind of 1.5 km/h.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `CITY_COORDINATES` — the fixed set of known cities; add more by looking up their lat/long
- The commented MCP block at the bottom — swap in a real `mcp_servers` URL and beta flag to try it live

### See also

- `../code_interpreter/README.md`, `../web_search/README.md` — the two fully server-side (Anthropic-hosted) tools this template's MCP reference section contrasts with
- `../../tool_registry/README.md` — a catalog pattern worth combining with this once you have more than one external API connector
