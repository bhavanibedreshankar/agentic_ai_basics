"""
CONCEPT: RAG (Retrieval-Augmented Generation) — dynamically fetching
relevant documents from a knowledge base and injecting them into the
prompt BEFORE generating a response, so the model answers grounded in
that content instead of (or in addition to) what it memorized during
training.

This template implements the CLASSIC RAG pattern — the one the term
originally described: retrieve, then generate, every single time, with no
model decision in between. Compare this with
../embedding/embedding_search.py and ../../Task_and_State_Management/context_management/retrieval.py,
where retrieval is a TOOL Claude decides whether and when to call. Both
are valid architectures:

  - Classic RAG (this file): simpler, deterministic, works even without
    tool-use support at all. Always pays the retrieval cost, even for
    queries that didn't need it (e.g. "thanks!" still triggers a search).
  - Tool-based retrieval: the model can skip retrieval when it's not
    needed, and can search multiple times with different queries within
    one turn — more flexible, at the cost of relying on the model to
    decide correctly.

The full pipeline demonstrated here: CHUNK documents -> EMBED each chunk
-> given a query, EMBED the query -> RETRIEVE the top-k most similar
chunks -> GENERATE an answer with those chunks injected as context. This
is the same embed()/cosine_similarity() mechanic as
../embedding/embedding_search.py and the same chunking idea as
../chunking/chunking_strategies.py, wired together end to end.

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the "
    "context provided below. If the context doesn't contain the answer, "
    "say so explicitly rather than guessing or using outside knowledge."
)

# ---------------------------------------------------------------------------
# Documents to index — a small internal knowledge base, the classic RAG
# demo use case (an internal docs Q&A bot).
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "pto-policy": (
        "Employees accrue 15 days of paid time off (PTO) per year, "
        "credited monthly at 1.25 days. PTO requests must be submitted "
        "through the HR portal at least 5 business days in advance for "
        "requests longer than 2 days. Unused PTO carries over up to a "
        "maximum of 5 days into the following year; anything beyond that "
        "is forfeited."
    ),
    "expense-reimbursement": (
        "Business expenses under $50 can be submitted with just a "
        "receipt photo through the expense app. Expenses over $50 "
        "require manager approval before submission. Reimbursements are "
        "processed within 10 business days and paid out with the next "
        "payroll cycle. Meals during business travel are reimbursed up "
        "to $75/day."
    ),
    "remote-work-guidelines": (
        "Employees may work remotely up to 3 days per week without prior "
        "approval. Fully remote arrangements require manager and HR "
        "sign-off. Remote employees must be reachable during core hours "
        "(10am-3pm in their local time zone) and attend all-hands "
        "meetings via video."
    ),
    "onboarding-checklist": (
        "New hires complete IT setup (laptop, accounts, VPN) on day one. "
        "Benefits enrollment must be completed within the first 30 days. "
        "A 30-60-90 day check-in is scheduled with the new hire's manager "
        "to review progress. Required compliance training must be "
        "finished within the first two weeks."
    ),
}

# ---------------------------------------------------------------------------
# CHUNKING — see ../chunking/chunking_strategies.py for a deeper look at
# strategies and trade-offs. These documents are short enough that each
# one is already close to a good chunk size, so we chunk by sentence with
# a generous max size — mostly this keeps the pipeline honest (chunking
# always happens before embedding) rather than doing heavy splitting.
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chunk_size: int = 220) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chunk_size and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# EMBEDDING — identical mechanic to ../embedding/embedding_search.py. See
# that file's comments for the full explanation of what this is a
# simplified stand-in for (a real embeddings API) and why.
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "he", "she", "what", "when", "where",
    "why", "which", "let", "lets", "can", "will", "must",
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
# BUILD THE INDEX — chunk every document, then embed every chunk. This
# happens ONCE, up front, not per query.
# ---------------------------------------------------------------------------
INDEX: list[tuple[str, str, list[float]]] = []
for doc_id, doc_text in DOCUMENTS.items():
    for chunk in chunk_text(doc_text):
        INDEX.append((doc_id, chunk, embed(chunk)))


# ---------------------------------------------------------------------------
# RETRIEVE
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Embed the query and rank every chunk in the index by similarity."""
    query_vector = embed(query)
    scored = [(doc_id, chunk, cosine_similarity(query_vector, vec)) for doc_id, chunk, vec in INDEX]
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# GENERATE — the "G" in RAG. A single, direct API call with retrieved
# context injected into the prompt. Notice there's no `tools=` parameter
# and no agentic loop here at all — retrieval already happened in Python,
# BEFORE this call, so Claude doesn't need to ask for it.
# ---------------------------------------------------------------------------
def answer_query(query: str) -> str:
    retrieved = retrieve(query)

    print(f"  [retrieved {len(retrieved)} chunks]")
    for doc_id, chunk, score in retrieved:
        print(f"    [{doc_id}] (similarity: {score:.3f}) {chunk[:70]}...")

    context = "\n\n".join(f"[Source: {doc_id}]\n{chunk}" for doc_id, chunk, _ in retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=RAG_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Internal docs assistant — {len(DOCUMENTS)} documents chunked into {len(INDEX)} indexed pieces.")
    print("Type 'exit' to end the conversation.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue

        answer = answer_query(query)
        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()
