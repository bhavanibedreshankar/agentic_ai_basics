"""
CONCEPT: External memory / vector store — long-term facts stored OUTSIDE
the conversation, in a persistent store, retrieved by semantic similarity
rather than always being resent in full.

This combines two things each already covered separately elsewhere in
this repo, into the pattern a real vector database gives you:
  - PERSISTENCE, like ../memory_management/basic_agentic_memory.py —
    facts survive after the process exits, written to disk.
  - SEMANTIC RETRIEVAL, like ../../RAG_and_Knowledge/embedding/ — facts
    are embedded into vectors and found by cosine similarity, not just
    injected wholesale.

The difference from both: ../memory_management/basic_agentic_memory.py
injects EVERY saved fact into the system prompt on every single call —
fine for a handful of facts, but that stops scaling once there are
hundreds of them. ../../RAG_and_Knowledge/embedding/embedding_search.py's
knowledge base is READ-ONLY, fixed at startup. This template's store is
WRITABLE at runtime (a save_memory tool) and only ever retrieves the
few entries relevant to the current query (a search_memory tool) — the
actual shape of a Pinecone, Chroma, or Weaviate integration, just backed
by a JSON file with an embed()/cosine_similarity() pair instead of a real
vector database (see ../../RAG_and_Knowledge/embedding/README.md for what
that simplification trades away).

Type 'exit' to end the conversation. Saved facts persist in
external_memory.json and are still there next time you run this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a helpful assistant with a persistent external memory. Use "
    "save_memory whenever the user shares a fact worth remembering "
    "long-term. Use search_memory to check whether you already know "
    "something relevant before answering, or when the user asks what you "
    "remember."
)

STORE_FILE = Path(__file__).parent / "external_memory.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# EMBEDDING — same mechanic as ../../RAG_and_Knowledge/embedding/embedding_search.py.
# See that file's comments for the full explanation of this simplification
# and its known limitations.
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "what", "when", "where", "why",
}


def embed(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for raw_word in text.lower().split():
        word = raw_word.strip(".,!?()`:;\"'")
        if not word or word in _STOPWORDS:
            continue
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[bucket] += 1.0
    magnitude = math.sqrt(sum(v * v for v in vector))
    return vector if magnitude == 0 else [v / magnitude for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# THE STORE — a flat JSON file of {id, text, vector} records, standing in
# for a real vector database's collection. Loaded fresh and saved back
# after every write, so it's always current on disk.
# ---------------------------------------------------------------------------
def _load_store() -> list[dict]:
    if not STORE_FILE.exists():
        return []
    return json.loads(STORE_FILE.read_text())


def _save_store(records: list[dict]) -> None:
    STORE_FILE.write_text(json.dumps(records, indent=2))


def save_memory(fact: str) -> str:
    """CONCEPT: writing to external memory. Embed once, at write time —
    same principle as ../../RAG_and_Knowledge/embedding/'s "build the
    index once, up front": the expensive step (computing a vector)
    happens exactly once per fact, not on every future search.
    """
    records = _load_store()
    record = {"id": uuid.uuid4().hex[:8], "text": fact, "vector": embed(fact)}
    records.append(record)
    _save_store(records)
    return f"Saved to external memory: {fact}"


def search_memory(query: str, top_k: int = 3) -> str:
    """CONCEPT: reading from external memory via similarity, not a full
    dump. Only the facts relevant to THIS query get pulled in — this is
    what lets the store scale to thousands of facts without every one of
    them bloating every request the way
    ../memory_management/basic_agentic_memory.py's wholesale system-
    prompt injection would.
    """
    records = _load_store()
    if not records:
        return "No memories saved yet."

    query_vector = embed(query)
    scored = [(r["text"], cosine_similarity(query_vector, r["vector"])) for r in records]
    scored.sort(key=lambda item: item[1], reverse=True)
    top_matches = [text for text, score in scored[:top_k] if score > 0]

    if not top_matches:
        return "No relevant memories found for this query."
    return "\n".join(f"- {text}" for text in top_matches)


TOOLS = [
    {
        "name": "save_memory",
        "description": "Save a fact to persistent external memory, to be recalled in future conversations via semantic search.",
        "input_schema": {
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "The fact to remember"}},
            "required": ["fact"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search external memory for facts relevant to a query, ranked by semantic similarity.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "save_memory":
        return save_memory(**tool_input), False
    if name == "search_memory":
        return search_memory(**tool_input), False
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

    existing = len(_load_store())
    print(f"External memory assistant — {existing} facts already stored.")
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
        run_turn(messages)


if __name__ == "__main__":
    main()
