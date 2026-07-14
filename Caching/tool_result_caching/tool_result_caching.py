"""
CONCEPT: Tool-result caching -- memoizing the RETURN VALUE of an
expensive, idempotent tool call in a plain client-side data structure, so
calling it again with the same arguments returns instantly instead of
re-running the underlying work.

This is a different layer from ../context_caching/context_caching.py.
Context caching is a SERVER-SIDE mechanism that caches how the model
processed a prompt PREFIX (via `cache_control`) -- it never touches your
tool's return values, and Claude still has to decide to call the tool
and read its result either way. Tool-result caching is entirely
CLIENT-SIDE: a dictionary that intercepts the call to your own tool
function before it ever reaches the (slow, or rate-limited, or metered)
work behind it. The two are complementary -- a real agent typically wants
both.

This template deliberately keeps the cache UNBOUNDED and permanent for
the life of the process: once a (tool, arguments) pair has been seen, its
result is cached forever, with no size limit and no expiration. That's a
reasonable choice for something like a static lookup, but it's also a
trap for anything whose answer can go stale (stock prices, weather,
account balances) or whose key space is unbounded (a cache entry per
user, growing without limit). ../cache_eviction_policies/cache_eviction_policies.py
picks up exactly where this template stops: the same idea, but bounded
by size (LRU eviction) and by time (TTL expiration).

Use case: a travel-planning agent with a get_weather(city) tool that
simulates a slow, rate-limited network call (a real weather API would
have similar latency and quota concerns). Asking about the same city
twice in one conversation should hit the network only once.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys
import time

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a travel-planning assistant. Use the get_weather tool whenever a "
    "question depends on a city's current weather, rather than guessing."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current simulated weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]

# A fixed set of canned conditions, so results are deterministic across a
# run -- this stands in for a real weather API's response.
_CONDITIONS = {
    "tokyo": "72F, partly cloudy",
    "lisbon": "68F, clear skies",
    "reykjavik": "41F, light rain",
}

# ---------------------------------------------------------------------------
# CONCEPT: the cache. Keyed on the exact arguments a call was made with --
# here just `city`, lowercased so "Tokyo" and "tokyo" share an entry. An
# unbounded dict is the simplest possible cache: no eviction, no expiry,
# entries live as long as the process does.
# ---------------------------------------------------------------------------
_weather_cache: dict[str, str] = {}
cache_hits = 0
cache_misses = 0


def _fetch_weather_from_network(city: str) -> str:
    """The expensive part: a simulated slow network call. In a real tool
    this would be an HTTP request with real latency and a real API quota
    -- both good reasons not to repeat it for a question you've already
    answered this session.
    """
    time.sleep(0.3)  # stands in for real network latency
    condition = _CONDITIONS.get(city.lower())
    if condition is None:
        return f"No weather data available for '{city}'."
    return f"{city.title()}: {condition}"


def get_weather(city: str) -> str:
    """CONCEPT: the memoization wrapper. Checks the cache BEFORE doing any
    expensive work, and populates it AFTER -- the same shape as
    functools.lru_cache, written out by hand so the hit/miss bookkeeping
    is visible rather than hidden behind a decorator.
    """
    global cache_hits, cache_misses
    key = city.strip().lower()
    if key in _weather_cache:
        cache_hits += 1
        print(f"    [cache hit  for '{city}' -- skipping network call]")
        return _weather_cache[key]

    cache_misses += 1
    print(f"    [cache miss for '{city}' -- fetching...]")
    result = _fetch_weather_from_network(city)
    _weather_cache[key] = result
    return result


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "get_weather":
        return get_weather(tool_input["city"])
    return f"Unknown tool: {name}"


def run_turn(messages: list[dict]) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        output_config={"effort": EFFORT},
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        assistant_blocks = [b.model_dump() for b in response.content]
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text = execute_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )

    reply = "".join(block.text for block in response.content if block.type == "text")
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
    return reply


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Travel-planning assistant. Type 'exit' to end and see cache hit/miss totals.")
    print("Try: \"What's the weather in Tokyo?\" then later \"Should I pack a coat for Tokyo?\"\n")

    messages: list[dict] = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print(f"\n--- Cache summary: {cache_hits} hits / {cache_misses} misses ---")
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        reply = run_turn(messages)
        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
