"""
CONCEPT: Embedding — converting text into a vector (a list of numbers) so
that semantically similar text can be found via math instead of exact
string or keyword matching.

Real embedding models (Anthropic recommends Voyage AI; OpenAI's
text-embedding-3, or a local sentence-transformers model, are common
alternatives — the Claude API itself doesn't produce embeddings) use a
neural network trained on huge amounts of text to place similar meanings
near each other in vector space: "dog" and "puppy" end up close together
even though they share no letters, because the model learned they're used
in similar contexts.

This template's `embed()` function is a simplified, DEPENDENCY-FREE stand-in
that captures word OVERLAP, not true meaning — it hashes each word into one
of a fixed number of buckets and counts occurrences ("feature hashing").
That's enough to demonstrate exactly how vector search WORKS — build a
vector for each document, build a vector for the query, compare vectors —
without needing an embeddings API key or downloading a model. Swap embed()
out for a real embeddings API call and everything else in this file
(cosine_similarity, search) keeps working completely unchanged, because
they only care that embed() returns a fixed-length list of floats.

Contrast with ../../Task_and_State_Management/context_management/retrieval.py: that template scored
relevance by counting shared words directly (bag-of-words overlap). This
template does something related in spirit but represents each document as
a persistent VECTOR you compute once and re-use, and ranks by a proper
similarity metric (cosine similarity) — the actual mechanic real vector
databases use, just at a much smaller scale.

Use case: a documentation assistant, same knowledge base shape as
../../Task_and_State_Management/context_management/retrieval.py, but backed by vector search instead
of a keyword search tool. Type 'exit' to end the conversation.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a documentation assistant. You have access to a "
    "search_knowledge_base tool backed by vector search — use it whenever "
    "a question might be covered by the notes, rather than answering from "
    "general knowledge. If it finds nothing relevant, say so honestly."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: turning text into a vector
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64  # a real embedding model might use 384, 1024, or more dims

# Common function words carry almost no topical signal but show up in
# nearly every sentence — left in, they add noise that can drown out the
# few words that actually distinguish one document from another. Real
# embedding models learn to downweight these automatically; this simple
# hashing scheme has to be told explicitly.
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "he", "she", "what", "when", "where",
    "why", "which", "let", "lets", "can", "will", "use", "used", "using",
}


def embed(text: str) -> list[float]:
    """Convert text into a fixed-length vector of floats.

    Each word is hashed into one of EMBEDDING_DIM "buckets" and increments
    that bucket's count — conceptually similar to a real embedding's
    dimensions, just far simpler (each dimension here roughly means "how
    many words landed in this bucket" rather than a learned semantic
    feature). We use hashlib rather than Python's built-in hash() because
    hash() is randomized per process for strings — hashlib gives the same
    bucket for the same word every time, which we need for consistent
    results across runs.

    LIMITATION worth being honest about: this only captures word overlap,
    not meaning — "purchase" and "buy" hash to unrelated buckets even
    though they mean the same thing, so a query using different wording
    than the source document can still score poorly. A real embedding
    model is trained specifically to avoid that failure mode; this one
    can't.
    """
    vector = [0.0] * EMBEDDING_DIM
    for raw_word in text.lower().split():
        word = raw_word.strip(".,!?()`:;\"'")
        if not word or word in _STOPWORDS:
            continue
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[bucket] += 1.0

    # L2-normalize to unit length. This does two things: it makes cosine
    # similarity reduce to a plain dot product (cheaper to compute), and it
    # stops longer documents from scoring higher purely because they have
    # more words — only the DIRECTION of the vector matters, not its size.
    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0:
        return vector
    return [v / magnitude for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """CONCEPT: measuring vector similarity. Cosine similarity ranges from
    -1 (opposite direction) to 1 (identical direction). For unit-length
    vectors (like the ones embed() produces) it's just the dot product —
    this single operation, repeated against every vector in an index, is
    what every vector database is ultimately built around, just optimized
    for millions of vectors instead of five.
    """
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# CONCEPT: building the index — do this ONCE, up front
# ---------------------------------------------------------------------------
# In a real system, embedding every document is the expensive part (an API
# call per document, or per chunk). You embed once when a document is
# added to the knowledge base and store the resulting vector — you never
# re-embed a document just to answer a query.
KNOWLEDGE_BASE = {
    "python-typing": (
        "Python's type hints (PEP 484) let you annotate variables and function "
        "signatures with expected types, e.g. `def add(a: int, b: int) -> int`. "
        "They're optional and unenforced at runtime — tools like mypy check "
        "them statically before you ever run the code."
    ),
    "git-basics": (
        "Git tracks changes via commits, each a snapshot of the repository. "
        "`git add` stages changes, `git commit` records them, and `git push` "
        "sends commits to a remote. Branches let you develop features in "
        "isolation before merging them back."
    ),
    "rest-apis": (
        "A REST API exposes resources over HTTP using standard verbs: GET to "
        "read, POST to create, PUT/PATCH to update, DELETE to remove. "
        "Responses are typically JSON, and status codes (200, 404, 500) "
        "indicate the outcome of the request."
    ),
    "docker-containers": (
        "Docker packages an application with its dependencies into a "
        "container — an isolated, portable unit that runs the same way on "
        "any machine. Images are built from a Dockerfile and run as "
        "containers via `docker run`."
    ),
    "regular-expressions": (
        "Regular expressions describe text patterns for searching and "
        "matching. `\\d+` matches one or more digits, `.*` matches anything, "
        "and `^`/`$` anchor to the start or end of a line. Python's `re` "
        "module implements them."
    ),
}

# The index: precomputed (doc_id, text, vector) triples. This is a stand-in
# for a real vector database (e.g. a managed service, or a local library
# like FAISS/Chroma) — those add fast approximate search over millions of
# vectors; the underlying idea (store vectors, compare with cosine
# similarity) is exactly what's happening here at a scale of five.
INDEX: list[tuple[str, str, list[float]]] = [
    (doc_id, text, embed(text)) for doc_id, text in KNOWLEDGE_BASE.items()
]


def vector_search(query: str, top_k: int = 2) -> list[tuple[str, str, float]]:
    """Embed the query, score it against every vector in the index, and
    return the top_k matches as (doc_id, text, score) — ranked highest
    similarity first.
    """
    query_vector = embed(query)
    scored = [(doc_id, text, cosine_similarity(query_vector, vec)) for doc_id, text, vec in INDEX]
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


def search_knowledge_base(query: str) -> str:
    """The tool handler Claude calls. Runs vector_search and formats the
    results as text, including similarity scores so the ranking is
    visible — helpful for understanding why one result outranked another.
    """
    results = vector_search(query)
    print(f"  [vector search: top {len(results)} of {len(INDEX)} documents]")
    for doc_id, _, score in results:
        print(f"    {doc_id}: {score:.3f}")

    if not results or results[0][2] <= 0:
        return "No relevant documents found."
    return "\n\n".join(f"[{doc_id}] (similarity: {score:.3f}) {text}" for doc_id, text, score in results)


TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the documentation knowledge base using semantic vector "
            "search and return the most relevant entries. Call this before "
            "answering questions about Python typing, Git, REST APIs, "
            "Docker, or regular expressions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "search_knowledge_base":
        return search_knowledge_base(**tool_input), False
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

    print(f"Documentation assistant — {len(INDEX)} documents indexed (vector search demo).")
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
