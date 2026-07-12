"""
CONCEPT: In-context memory — information held within the active
conversation's `messages` list. This is the memory every template in
../../agentic_loop/ and beyond has been using implicitly; this template
makes its two defining properties impossible to miss:

  1. EPHEMERAL — it lives only in this Python process's `messages`
     variable. Nothing here is written to disk. Stop the script and it's
     gone completely, unlike ../semantic_memory/ or
     ../episodic_memory/, which persist to a file specifically so they
     survive that.
  2. LIMITED BY TOKEN SIZE — every message ever sent in this conversation
     gets resent, in full, on every single API call (see
     ../../agentic_loop/README.md). That's cheap at first and expensive
     later — and eventually impossible, since context windows have a
     hard maximum.

Real context windows are huge (1M tokens on current models) — too big to
usefully demonstrate hitting the limit in a terminal demo. This template
imposes an artificial, tiny MAX_CONTEXT_TOKENS instead, so you can
actually watch eviction happen after a handful of messages, using the
same real count_tokens endpoint from ../../token_tracking/ to measure
usage before every call. When the limit is exceeded, the OLDEST
user/assistant pair is evicted before sending — the simplest possible
eviction policy, deliberately dumber than
../../context_management/pruning.py's selective pruning or
../../context_management/summarization.py's summarize-then-replace. The
point here isn't a good eviction strategy — those templates cover that —
it's showing the raw problem those strategies exist to solve.

Ask about something, keep chatting until eviction happens, then ask about
the thing you mentioned first — the agent will have genuinely forgotten
it, with nothing left to recover it from. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant. Answer questions on any topic clearly and concisely."

# Deliberately tiny — real context windows are ~1M tokens on current
# models. This is small enough to hit within a few short turns of an
# interactive demo, so eviction is actually observable.
MAX_CONTEXT_TOKENS = 500

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def count_tokens(messages: list[dict]) -> int:
    """The real count_tokens endpoint (see ../../token_tracking/) — not a
    guess from len(messages) or character counts.
    """
    result = client.messages.count_tokens(model=MODEL, system=SYSTEM_PROMPT, messages=messages)
    return result.input_tokens


def evict_oldest_pair(messages: list[dict]) -> list[dict]:
    """CONCEPT: the simplest possible eviction policy — drop the oldest
    user/assistant pair (first two messages) and never look at them
    again. No summarizing, no selective pruning: once evicted here,
    that information is unrecoverable — genuinely, not just
    inconvenient to access. That's the point: this is the failure mode
    ../../context_management/'s techniques exist to avoid.
    """
    return messages[2:] if len(messages) >= 2 else messages


def ask_claude(messages: list[dict]) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"In-context memory demo (limit: {MAX_CONTEXT_TOKENS} tokens — artificially tiny). Type 'exit' to quit.\n")
    print("Mention something early on, then keep chatting until eviction happens, then ask about it again.\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! (and everything above is gone now too — nothing here was ever saved)")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # CONCEPT: check BEFORE sending, evict if needed, so every call
        # actually respects the limit rather than just reacting after
        # the fact.
        while count_tokens(messages) > MAX_CONTEXT_TOKENS and len(messages) > 2:
            evicted = messages[:2]
            messages = evict_oldest_pair(messages)
            preview = str(evicted[0].get("content", ""))[:60]
            print(f"  [evicted oldest turn to stay under {MAX_CONTEXT_TOKENS} tokens: \"{preview}...\"]")

        tokens_used = count_tokens(messages)
        print(f"  [context: {tokens_used}/{MAX_CONTEXT_TOKENS} tokens, {len(messages)} messages]")

        reply = ask_claude(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
