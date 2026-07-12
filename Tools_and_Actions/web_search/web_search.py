"""
CONCEPT: Web Search Tool — lets the agent retrieve live information from
the internet, past its training cutoff, without you building or hosting
any search backend at all.

Like `../code_interpreter/`, this is a SERVER-SIDE tool: declare
`{"type": "web_search_20260209", "name": "web_search"}` and Claude
issues search queries, fetches results, and reads them — all on
Anthropic's infrastructure. Contrast with every custom search tool built
elsewhere in this repo (`../../context_management/retrieval.py`,
`../../RAG_and_Knowledge/embedding/`, `../../Memory/external_memory/`) —
those all search a small, fixed, LOCAL knowledge base you defined ahead
of time. This tool searches the actual, live, current internet. There's
no knowledge base to write, no embedding function to implement, and no
execute_tool() dispatch — same shape as `../code_interpreter/`, and for
the same reason: nothing here runs on your machine.

Claude also cites its sources automatically when this tool is used — the
response's text blocks come back annotated with which search result each
claim is based on, which this template prints out separately from the
answer so the sourcing is visible.

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
SYSTEM_PROMPT = (
    "You are a research assistant with access to live web search. Use it "
    "for anything time-sensitive or recent — current events, current "
    "prices, latest versions of software, anything your training data "
    "might be outdated on. Don't guess when you can look it up."
)

# max_uses caps how many searches Claude can run in a single turn — a
# real cost/latency control, since each search is billed. allowed_domains
# / blocked_domains (not used here) can scope results to trusted sources.
TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def print_response_content(response) -> None:
    """Print search queries as they're issued, then the answer, then a
    distinct citations section — so it's clear WHAT Claude searched for
    and WHERE the final claims actually came from, not just the answer
    text on its own.
    """
    citations_seen = []

    for block in response.content:
        if block.type == "server_tool_use" and block.name == "web_search":
            query = block.input.get("query", "")
            print(f"\n  [searching] {query}")
        elif block.type == "web_search_tool_result":
            # .content is a list on success, or a single error object on
            # failure (e.g. max_uses_exceeded) — the API returns HTTP 200
            # either way, so check the shape rather than assuming success.
            if isinstance(block.content, list):
                print(f"  [found {len(block.content)} results]")
            else:
                print(f"  [search error: {getattr(block.content, 'error_code', block.content)}]")
        elif block.type == "text":
            print(f"\nClaude: {block.text}")
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                title = getattr(citation, "title", None)
                if url:
                    citations_seen.append((title or url, url))

    if citations_seen:
        print("\n  Sources:")
        for title, url in citations_seen:
            print(f"    - {title}: {url}")


def run_turn(messages: list[dict]) -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        output_config={"effort": EFFORT},
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})
    print_response_content(response)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Web search assistant. Type 'exit' to end the conversation.\n")
    print("Try: \"What's the latest stable version of Python?\"\n")

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
        print()


if __name__ == "__main__":
    main()
