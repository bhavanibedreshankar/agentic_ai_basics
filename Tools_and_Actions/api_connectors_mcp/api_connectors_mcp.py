"""
CONCEPT: API Connectors / MCP — integrating an agent with an external
service (a database, a SaaS app, a public API) either by wrapping a REST
call as a custom tool yourself, or by connecting to a Model Context
Protocol (MCP) server that already exposes one.

This template runs the FIRST approach end to end: a custom tool that
calls a real, live, public REST API (Open-Meteo — free, no API key) using
nothing but Python's standard library (`urllib.request`), following the
exact same tool-definition/execute_tool shape as every custom tool in
this repo (`../tool_use/`, `../../Agent_Frameworks_and_Patterns/tool_registry/`). The only thing new
here is that the tool's implementation makes a real network call instead
of computing something locally or reading a local file.

The SECOND approach — MCP — is shown as a commented, NOT-executed
reference at the bottom of this file, rather than run live. MCP servers
require a real server URL and often authentication (see
`shared/managed-agents-tools.md` → Vaults if you're integrating one for
real); guessing a URL to demonstrate against would mean depending on
infrastructure this template doesn't control and can't verify stays
available. The commented block shows the exact shape: `mcp_servers` on
the request declares the server, and a `mcp_toolset` entry in `tools`
grants access to it — no execute_tool() needed for MCP tools either,
since (like `../code_interpreter/` and `../web_search/`) Claude calls
them directly; the MCP server runs on its own infrastructure, not yours.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a weather assistant. Use get_weather to look up current "
    "conditions for a location rather than guessing — you have no other "
    "source of live weather data."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# A small, fixed set of cities mapped to coordinates, since the free
# geocoding step is a whole separate API call — keeping this template
# focused on "wrap one external API as a tool" rather than "chain two
# APIs together".
CITY_COORDINATES = {
    "london": (51.51, -0.13),
    "new york": (40.71, -74.01),
    "tokyo": (35.68, 139.69),
    "berlin": (52.52, 13.41),
    "sydney": (-33.87, 151.21),
    "san francisco": (37.77, -122.42),
}


def get_weather(city: str) -> str:
    """CONCEPT: the API connector itself — a plain Python function that
    calls a real, external, unauthenticated REST API and returns a
    result. No SDK, no API key: `urllib.request` (standard library) is
    enough for a simple GET request. A connector for an authenticated
    service would add an Authorization header here; the shape of the
    function stays the same either way.
    """
    key = city.strip().lower()
    if key not in CITY_COORDINATES:
        return f"Unknown city '{city}'. Known cities: {', '.join(CITY_COORDINATES)}"

    lat, lon = CITY_COORDINATES[key]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
    except urllib.error.URLError as exc:
        return f"Error reaching weather API: {exc}"

    current = data.get("current", {})
    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")
    return f"{city.title()}: {temp}°C, wind {wind} km/h"


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather conditions for a known city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": f"One of: {', '.join(CITY_COORDINATES)}",
                },
            },
            "required": ["city"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "get_weather":
        return get_weather(**tool_input), False
    return f"Unknown tool: {name}", True


def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [result] {result_text}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Weather assistant (API connector demo). Type 'exit' to end the conversation.\n")
    print(f"Known cities: {', '.join(CITY_COORDINATES)}\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------
# REFERENCE ONLY — not executed. The MCP connector shape for comparison.
#
# Where a hand-written tool like get_weather() above calls the external
# API itself, MCP flips that: an MCP SERVER (run by you, or by whoever
# operates the service) exposes tools that Claude calls directly, over
# the network — same "server-side, no execute_tool()" shape as
# ../code_interpreter/ and ../web_search/, but for a service YOU choose
# rather than one Anthropic hosts. Two things are required together, on
# a beta endpoint:
#
#   response = client.beta.messages.create(
#       model="claude-sonnet-5",
#       max_tokens=4096,
#       betas=["mcp-client-2025-11-20"],
#       mcp_servers=[
#           {"type": "url", "url": "https://your-mcp-server.example.com/sse", "name": "my-service"},
#       ],
#       tools=[
#           {"type": "mcp_toolset", "mcp_server_name": "my-service"},
#       ],
#       messages=[{"role": "user", "content": "..."}],
#   )
#
# `mcp_servers` declares the connection; the `mcp_toolset` entry in
# `tools` is what actually grants Claude access to that server's tools —
# omitting it is a validation error even if `mcp_servers` is set. An
# authenticated server would need an `authorization_token` field on the
# server entry, or (for Managed Agents sessions specifically) a vault
# credential rather than a token in the request itself.
# -----------------------------------------------------------------------
