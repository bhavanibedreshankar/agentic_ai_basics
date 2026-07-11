"""
CONCEPT: Token tracking — how to measure and monitor token usage (and
therefore cost) in a Claude API application.

Every API response includes a `usage` object reporting exactly how many
tokens were consumed. This template wraps the basic chat loop from
basic_agentic_loop.py with per-turn and cumulative token tracking, plus a
pre-flight estimate using the count_tokens endpoint.

Concepts demonstrated:
  1. READING USAGE FROM A RESPONSE — response.usage.input_tokens /
     .output_tokens, and the cache-related fields that report cost savings
     when prompt caching is in play.
  2. PRE-FLIGHT TOKEN COUNTING — using client.messages.count_tokens() to see
     the size of a request BEFORE sending it, e.g. to warn on an unusually
     large request or decide whether to trim history first.
  3. CUMULATIVE TRACKING — accumulating usage across every turn of a
     conversation. Each API call in a multi-turn chat resends the full
     history (see basic_agentic_loop.py), so cost compounds as a
     conversation grows — this template makes that growth visible.
  4. COST ESTIMATION — converting token counts into an approximate dollar
     figure using the model's per-token pricing.

Type 'exit' to end the conversation and see a session usage summary.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant. Answer questions on any topic clearly and concisely."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: Pricing — token counts only matter because they translate into
# real cost. These are approximate standard rates (USD per 1 million
# tokens) for MODEL; check platform.claude.com/pricing for current numbers
# before relying on this for real budgeting — rates change, and some
# models have time-limited introductory pricing.
# ---------------------------------------------------------------------------
PRICE_PER_MILLION_INPUT = 3.00
PRICE_PER_MILLION_OUTPUT = 15.00


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT + (
        output_tokens / 1_000_000
    ) * PRICE_PER_MILLION_OUTPUT


class SessionUsage:
    """CONCEPT: cumulative tracking. A single response.usage only tells you
    about ONE API call. In a multi-turn conversation, every turn resends
    the full history, so cost compounds — this class adds up totals across
    the whole session so you can see the running picture, not just the
    latest turn.

    Note: this template doesn't trim history (unlike basic_agentic_memory.py's
    trim_history), so watch input_tokens climb turn over turn as the
    conversation grows — that growth is the whole point of tracking it.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def add(self, usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        # CONCEPT: cache fields. These report tokens served from (or
        # written to) the prompt cache, billed at a fraction of normal
        # input cost. They'll be 0 here since this template doesn't use
        # cache_control — but they're worth knowing: a production agent
        # with a large, stable system prompt would see cache_read climb
        # instead of input_tokens, cutting cost significantly.
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_input_tokens += usage.cache_read_input_tokens or 0
        self.turns += 1

    def summary(self) -> str:
        cost = estimate_cost(self.input_tokens, self.output_tokens)
        return (
            f"Turns:          {self.turns}\n"
            f"Input tokens:   {self.input_tokens:,}\n"
            f"Output tokens:  {self.output_tokens:,}\n"
            f"Cache reads:    {self.cache_read_input_tokens:,} (billed ~10x cheaper than input)\n"
            f"Cache writes:   {self.cache_creation_input_tokens:,}\n"
            f"Estimated cost: ${cost:.4f}"
        )


def preview_input_tokens(messages: list[dict]) -> int:
    """CONCEPT: pre-flight token counting. count_tokens() runs the request
    through the same tokenizer as a real call, WITHOUT generating a
    response — so it costs nothing to run and lets you check the size (and
    therefore cost) of a request before sending it. Useful for warning
    users on an oversized request, or deciding whether to trim history
    first.
    """
    count = client.messages.count_tokens(model=MODEL, system=SYSTEM_PROMPT, messages=messages)
    return count.input_tokens


def ask_claude(messages: list[dict], usage_tracker: SessionUsage) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=messages,
    )

    # CONCEPT: reading usage off the response. Every response carries a
    # `usage` object — this is the ground truth for what a call actually
    # cost, independent of any pre-flight estimate (the two numbers can
    # differ slightly, since count_tokens estimates the request only,
    # before the API applies its own exact accounting).
    usage_tracker.add(response.usage)
    turn_cost = estimate_cost(response.usage.input_tokens, response.usage.output_tokens)
    print(
        f"  [usage: {response.usage.input_tokens} in / "
        f"{response.usage.output_tokens} out, ~${turn_cost:.4f} this turn]"
    )

    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Chat with Claude. Type 'exit' to end and see a token usage summary.\n")

    messages: list[dict] = []
    usage_tracker = SessionUsage()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("\n--- Session summary ---")
            print(usage_tracker.summary())
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Pre-flight estimate before the real call, to demonstrate
        # count_tokens() directly — in a real app you'd use this number to
        # make a decision (warn, trim, switch model) rather than just
        # printing it.
        estimated_input = preview_input_tokens(messages)
        print(f"  [pre-flight estimate: ~{estimated_input} input tokens]")

        reply = ask_claude(messages, usage_tracker)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
