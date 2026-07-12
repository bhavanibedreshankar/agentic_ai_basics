"""
CONCEPT: Context summarization — instead of deleting old context (pruning)
or keeping everything forever (unbounded growth), COMPRESS it: use an
extra, focused LLM call to summarize older turns into a short paragraph,
then replace those turns with the summary. This keeps the GIST of what
was discussed while dramatically cutting token count — a middle ground
between pruning (loses detail entirely) and doing nothing (keeps
everything, cost grows without limit).

This is a hand-built version of the same idea behind Anthropic's built-in
"compaction" feature — worth understanding by hand even though a
production system would typically reach for the server-side feature
instead of reimplementing it.

The trade-off: summarization costs an EXTRA API call (time + tokens)
every time it runs, and it's lossy — anything not mentioned in the
summary is gone for good. It's worth doing when the alternative
(resending a huge, ever-growing history on every turn) costs more than
the occasional summarization call.

Use case: a general chat agent (like ../../Execution_Loops/agentic_loop/basic_agentic_loop.py) that, once
the conversation grows past SUMMARIZE_AFTER_TURNS messages, summarizes
everything except the most recent few turns and continues from there.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant. Answer questions on any topic clearly and concisely."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: when to summarize
# ---------------------------------------------------------------------------
# Once the history exceeds SUMMARIZE_AFTER_TURNS messages, everything
# except the most recent KEEP_RECENT_TURNS is collapsed into one summary
# message. Tune these based on how much recent detail matters for your use
# case versus how aggressively you want to cut token cost.
SUMMARIZE_AFTER_TURNS = 8
KEEP_RECENT_TURNS = 4

SUMMARY_SYSTEM_PROMPT = (
    "Summarize the following conversation between a user and an assistant "
    "in a few sentences, capturing any facts, decisions, or context that "
    "would matter for continuing the conversation later. Write in third "
    "person, e.g. 'The user asked about X; the assistant explained Y.'"
)


def format_transcript(messages: list[dict]) -> str:
    """Turn a list of {"role", "content"} messages into a flat, readable
    transcript string for the summarizer to read. This template only ever
    puts plain strings in `content` (see ask_claude/main below), so this
    stays simple — a version handling tool calls would need to also
    flatten tool_use/tool_result blocks into readable text.
    """
    lines = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            lines.append(f"{message['role']}: {content}")
    return "\n".join(lines)


def summarize(messages: list[dict]) -> str:
    """CONCEPT: the summarization call itself — a completely separate,
    narrowly-focused API call from the main conversation. Same idea as the
    chained steps in ../../prompt_chaining/basic_prompt_chaining.py: one job, its own system
    prompt, no shared history with the ongoing chat.
    """
    transcript = format_transcript(messages)
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SUMMARY_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": transcript}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def maybe_summarize(messages: list[dict]) -> list[dict]:
    """CONCEPT: replacing old turns with a summary. If the conversation has
    grown past the threshold, summarize everything except the most recent
    KEEP_RECENT_TURNS messages, and splice the result in as a SINGLE
    leading message — collapsing many messages down to one.

    Returns a new list; doesn't mutate the input in place, since we're
    replacing (not just trimming) part of it.
    """
    if len(messages) <= SUMMARIZE_AFTER_TURNS:
        return messages

    to_summarize = messages[:-KEEP_RECENT_TURNS]
    recent = messages[-KEEP_RECENT_TURNS:]

    summary_text = summarize(to_summarize)
    print(f"\n  [summarized {len(to_summarize)} earlier messages into 1 summary message]")
    print(f"  [summary: {summary_text}]\n")

    # A plain "user" message works regardless of what role `recent[0]` is —
    # the API allows consecutive same-role messages (it just combines
    # them), and starting with "user" always satisfies the "first message
    # must be user" rule.
    summary_message = {"role": "user", "content": f"(Summary of earlier conversation: {summary_text})"}
    return [summary_message] + recent


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

    print("Chat with Claude (context summarization demo). Type 'exit' to end the conversation.\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # CONCEPT: check (and possibly summarize) BEFORE the main call, so
        # the summarized-down history is what actually gets sent — not
        # just tracked for next time.
        messages = maybe_summarize(messages)

        reply = ask_claude(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
