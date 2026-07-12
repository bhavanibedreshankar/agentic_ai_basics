"""
CONCEPT: Episodic memory — records of past INTERACTIONS or task runs
(what was asked, what was done, when, how it went), used to inform future
behavior. This is memory about EVENTS, not facts.

That's the key distinction from the other memory types in this
directory:
  - ../external_memory/ and ../semantic_memory/ store WHAT IS TRUE — a
    fact or preference, timeless, no notion of "when this happened" or
    "what task this came from".
  - Episodic memory stores WHAT HAPPENED — a specific request, a specific
    response, a timestamp, and (optionally) whether it worked. "The user
    asked me to summarize a PDF on 2026-03-01" is an episode.
    "The user prefers concise summaries" is a fact you might extract
    FROM that episode, but it's a different kind of memory once
    extracted — that extraction step is deliberately NOT done here to
    keep this template focused on the episodic layer alone.

Episodes here are logged AUTOMATICALLY by the program after every
completed turn — logging isn't something the agent decides to do (unlike
saving a fact, which the model chooses via a tool call in
../external_memory/). Recall IS a deliberate choice: a recall_episodes
tool the agent can use to check "have I handled something like this
before, and how did it go?" before tackling a new request.

Type 'exit' to end the conversation. Past episodes persist in
episodes.json and are available again next run.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a task assistant with a memory of past interactions. Before "
    "tackling a request that resembles something you may have done "
    "before, use recall_episodes to check your history — if a similar "
    "request went well before, follow the same approach; if it didn't, "
    "adjust."
)

EPISODES_FILE = Path(__file__).parent / "episodes.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def _load_episodes() -> list[dict]:
    if not EPISODES_FILE.exists():
        return []
    return json.loads(EPISODES_FILE.read_text())


def _save_episodes(episodes: list[dict]) -> None:
    EPISODES_FILE.write_text(json.dumps(episodes, indent=2))


def log_episode(user_request: str, agent_response: str) -> dict:
    """CONCEPT: automatic logging. Called by the program itself after
    every turn completes — not a tool the model invokes, because
    remembering that an interaction happened isn't a judgment call the
    way deciding a fact is worth saving (in ../external_memory/) is. Every
    completed turn becomes an episode, unconditionally.
    """
    episodes = _load_episodes()
    episode = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": user_request,
        "response": agent_response,
    }
    episodes.append(episode)
    _save_episodes(episodes)
    return episode


def recall_episodes(query: str, top_k: int = 3) -> str:
    """CONCEPT: recalling past episodes by keyword overlap — a different,
    simpler retrieval technique than ../external_memory/'s embeddings, to
    show that episodic recall doesn't require the same machinery: past
    task descriptions tend to share literal words with a new, similar
    request ("summarize a PDF" vs. "summarize this document"), so a
    simpler match is often good enough here.
    """
    episodes = _load_episodes()
    if not episodes:
        return "No past episodes recorded yet."

    query_words = set(query.lower().split())
    scored = []
    for ep in episodes:
        overlap = len(query_words & set(ep["request"].lower().split()))
        if overlap > 0:
            scored.append((overlap, ep))

    if not scored:
        return "No similar past episodes found."

    scored.sort(key=lambda item: item[0], reverse=True)
    top_matches = [ep for _, ep in scored[:top_k]]
    return "\n\n".join(
        f"[{ep['timestamp']}] Request: {ep['request']}\nResponse: {ep['response']}" for ep in top_matches
    )


TOOLS = [
    {
        "name": "recall_episodes",
        "description": "Search past interaction history for episodes similar to a given request, to see how similar requests were handled before.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A description of the current request to find similar past episodes for"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "recall_episodes":
        return recall_episodes(**tool_input), False
    return f"Unknown tool: {name}", True


def run_turn(user_input: str, messages: list[dict]) -> None:
    """Handles one user turn, including any recall_episodes tool calls,
    and logs the completed turn as a new episode once Claude has given
    its final response.
    """
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

        final_text = ""
        for block in response.content:
            if block.type == "text":
                final_text = block.text
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            if final_text:
                episode = log_episode(user_input, final_text)
                print(f"  [logged episode at {episode['timestamp']}]")
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
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

    existing = len(_load_episodes())
    print(f"Task assistant — {existing} past episodes recorded.")
    print("Type 'exit' to end the conversation.\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(user_input, messages)


if __name__ == "__main__":
    main()
