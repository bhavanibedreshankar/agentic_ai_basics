"""
CONCEPT: Context retrieval — instead of keeping everything in context
(short-term memory) or always injecting a fixed set of facts (long-term
memory, as in ../memory_management/basic_agentic_memory.py), store a LARGER pool of
information externally and pull in only the pieces relevant to the
CURRENT question, on demand.

This is the core idea behind RAG (retrieval-augmented generation): the
context window stays small and focused no matter how big the underlying
knowledge base gets, because only a relevant slice of it is ever loaded
in. Real systems typically use embeddings and a vector database to judge
relevance; this template uses simple keyword overlap instead, to stay
dependency-free while demonstrating the exact same pattern — a
search_notes TOOL the model calls to retrieve exactly what it needs,
rather than the caller stuffing everything into the prompt up front.

Use case: a documentation assistant with a small, fixed local knowledge
base of notes on a handful of unrelated topics. Instead of putting every
note into the system prompt on every request (wasteful, and dilutes
what's actually relevant), Claude calls search_notes(query) and only the
top-matching notes are pulled into context for that turn.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a documentation assistant. You have access to a search_notes "
    "tool over a local knowledge base — use it whenever a question might "
    "be covered by the notes, rather than answering from general "
    "knowledge. If search_notes finds nothing relevant, say so honestly."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the external knowledge store. This is deliberately BIGGER than
# what you'd want to paste into a system prompt every turn — the whole
# point of retrieval is that Claude never sees all of this at once, only
# whatever search_notes decides is relevant to the current query.
# ---------------------------------------------------------------------------
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


def search_notes(query: str, top_k: int = 2) -> str:
    """CONCEPT: relevance ranking. A real RAG system embeds the query and
    every document, then ranks by vector similarity. This does the same
    JOB with much simpler math: score each note by how many words it
    shares with the query, and return only the top_k highest-scoring
    notes — everything else in KNOWLEDGE_BASE stays out of context
    entirely for this turn.
    """
    query_words = set(query.lower().split())
    scored = []
    for topic, text in KNOWLEDGE_BASE.items():
        overlap = len(query_words & set(text.lower().split()))
        if overlap > 0:
            scored.append((overlap, topic, text))

    # The print below is here purely to make the retrieval effect visible
    # in this demo — a real tool handler wouldn't normally print.
    print(f"  [retrieval: matched {len(scored)} of {len(KNOWLEDGE_BASE)} notes in the knowledge base]")

    if not scored:
        return "No relevant notes found for this query."

    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(f"[{topic}] {text}" for _, topic, text in scored[:top_k])


TOOLS = [
    {
        "name": "search_notes",
        "description": (
            "Search a local knowledge base of technical notes and return the "
            "most relevant ones. Call this before answering questions about "
            "Python typing, Git, REST APIs, Docker, or regular expressions — "
            "don't rely on general knowledge for these topics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "search_notes":
        return search_notes(**tool_input), False
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

    print(f"Documentation assistant — {len(KNOWLEDGE_BASE)} notes in the knowledge base (context retrieval demo).")
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
