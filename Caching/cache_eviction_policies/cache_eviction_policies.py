"""
CONCEPT: Cache eviction policies -- once a cache is allowed to grow
(as in ../tool_result_caching/tool_result_caching.py's plain dict), something
eventually has to decide what gets THROWN OUT to keep it bounded. This
template replaces that unbounded dict with a BoundedCache that enforces
two independent limits:
  1. SIZE -- at most `maxsize` entries. Once full, adding a new entry
     evicts the Least Recently Used one (LRU) -- the entry that hasn't
     been read or written in the longest time, on the theory that it's
     the least likely to be asked for again soon.
  2. TIME -- each entry expires `ttl_seconds` after it was written (TTL:
     time-to-live), even if the cache never fills up. This matters for
     data that goes stale on its own, like weather or prices, regardless
     of how often it's accessed.

../tool_result_caching/tool_result_caching.py deliberately skipped both of
these -- its cache lives (and grows) for as long as the process runs, which
is fine for a short demo but wrong for a long-running agent serving many
distinct cities, accounts, or users: the cache would grow forever, and
entries would never reflect that the real-world answer might have changed.

Use case: the same simulated get_weather(city) tool as
../tool_result_caching/tool_result_caching.py, but now wrapped in a
BoundedCache with a small maxsize -- small enough that asking about more
cities than it can hold forces a visible eviction, and a short TTL --
short enough that re-asking about an old city after enough real time has
passed forces a fresh fetch instead of a stale cache hit.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict

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

_CONDITIONS = {
    "tokyo": "72F, partly cloudy",
    "lisbon": "68F, clear skies",
    "reykjavik": "41F, light rain",
    "nairobi": "78F, sunny",
    "oslo": "35F, snow flurries",
}


def _fetch_weather_from_network(city: str) -> str:
    time.sleep(0.3)  # stands in for real network latency
    condition = _CONDITIONS.get(city.lower())
    if condition is None:
        return f"No weather data available for '{city}'."
    return f"{city.title()}: {condition}"


class BoundedCache:
    """CONCEPT: a cache that enforces both a size limit and a time limit,
    instead of ../tool_result_caching/tool_result_caching.py's
    grow-forever dict.

    Implementation notes:
      - Backed by an OrderedDict, which remembers insertion/access order.
        `move_to_end` on every read or write keeps the LEAST recently
        used entry at the FRONT, so eviction is always "pop the front."
      - Each entry stores (value, written_at) so a read can check
        `now - written_at >= ttl_seconds` and treat an expired entry as a
        miss even though the key is technically still present.
    """

    def __init__(self, maxsize: int, ttl_seconds: float) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str) -> str | None:
        if key not in self._store:
            self.misses += 1
            return None
        value, written_at = self._store[key]
        if time.time() - written_at >= self.ttl_seconds:
            # CONCEPT: TTL expiration. The entry is still physically
            # present but treated as gone -- stale data is worse than no
            # data, so an expired hit is scored as a miss.
            del self._store[key]
            self.misses += 1
            self.expirations += 1
            return None
        # CONCEPT: LRU bookkeeping. Reading an entry marks it as
        # recently used by moving it to the back, so it won't be the
        # next thing evicted.
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.maxsize:
            # CONCEPT: LRU eviction. popitem(last=False) removes the
            # FRONT of the OrderedDict -- the least recently
            # read-or-written entry -- to make room for the new one.
            evicted_key, _ = self._store.popitem(last=False)
            self.evictions += 1
            print(f"    [evicted '{evicted_key}' to make room -- cache at maxsize={self.maxsize}]")
        self._store[key] = (value, time.time())

    def stats(self) -> str:
        return (
            f"hits={self.hits} misses={self.misses} "
            f"evictions={self.evictions} expirations={self.expirations} "
            f"current size={len(self._store)}/{self.maxsize}"
        )


# A small maxsize (3) and a short TTL (10s) so both eviction and
# expiration are easy to trigger within a short interactive session --
# a production cache would size these to its real traffic pattern.
weather_cache = BoundedCache(maxsize=3, ttl_seconds=10.0)


def get_weather(city: str) -> str:
    key = city.strip().lower()
    cached = weather_cache.get(key)
    if cached is not None:
        print(f"    [cache hit  for '{city}']")
        return cached
    print(f"    [cache miss for '{city}' -- fetching...]")
    result = _fetch_weather_from_network(city)
    weather_cache.put(key, result)
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

    print("Travel-planning assistant (bounded cache: maxsize=3, ttl=10s). Type 'exit' to end.")
    print("Try asking about 4+ different cities in a row (tokyo, lisbon, reykjavik, nairobi, oslo)")
    print("to see an eviction, then re-ask about the first one to see it get re-fetched.\n")

    messages: list[dict] = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print(f"\n--- Cache summary: {weather_cache.stats()} ---")
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        reply = run_turn(messages)
        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
