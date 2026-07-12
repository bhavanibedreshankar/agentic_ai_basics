"""
CONCEPT: Coarse-to-fine retrieval — a two-stage search funnel. First,
retrieve broadly (cheap, approximate) to narrow a large corpus down to a
small shortlist of promising candidates. Then, retrieve again within just
that shortlist at a finer grain (more precise, more expensive per item) to
pinpoint the specific relevant section.

Why not just search every fine-grained chunk directly, as
../rag/basic_rag.py does? At small scale (a handful of documents) that's
exactly the right call — it's simpler and there's no real cost problem to
solve. Coarse-to-fine earns its complexity at LARGER scale: if a corpus
has thousands of documents each split into dozens of chunks, embedding
and scoring every single chunk against every query is expensive. Coarse-
to-fine spends that expensive fine-grained work only on documents that
already looked promising at the cheap coarse pass — narrow first, then
spend precision where it counts.

STAGE 1 (coarse): embed each WHOLE document as one vector, rank all
documents, keep the top few.
STAGE 2 (fine): split ONLY those top documents into paragraph-level
chunks, embed and rank those chunks individually, to find the specific
section that answers the query.

This uses the same embed()/cosine_similarity() mechanic as
../embedding/embedding_search.py, and the same chunking idea as
../chunking/chunking_strategies.py — applied in two passes instead of one.

Run this file directly to see both stages, then a generated answer:

    python3 coarse_to_fine_retrieval.py
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

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the context provided "
    "below. If the context doesn't cover the question, say so explicitly."
)

# ---------------------------------------------------------------------------
# A handful of LONGER, multi-topic documents. Each one covers several
# distinct sections, so stage 1 must first pick the right DOCUMENT, and
# stage 2 must then pick the right SECTION within it.
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "employee-handbook": (
        "Compensation is reviewed annually each March, with adjustments "
        "effective the following pay period. Bonuses are discretionary "
        "and tied to both company and individual performance.\n\n"
        "Health insurance coverage begins on the first day of employment. "
        "The company covers 80% of premiums for employees and 50% for "
        "dependents. Open enrollment for plan changes runs each November.\n\n"
        "The dress code is business casual for office days and unrestricted "
        "for remote days. Client-facing meetings require business "
        "professional attire regardless of location."
    ),
    "engineering-handbook": (
        "Code review requires at least one approval before merging to "
        "main. Pull requests should be scoped to a single logical change "
        "and include a description of what changed and why.\n\n"
        "On-call rotations are weekly, Monday to Monday, with a handoff "
        "meeting each Monday morning. On-call engineers carry a pager and "
        "must acknowledge incidents within 15 minutes.\n\n"
        "Production deploys happen via the CI/CD pipeline only — direct "
        "manual deploys to production are disabled. Deploys require a "
        "passing test suite and are automatically rolled back on error "
        "rate spikes."
    ),
    "security-policy": (
        "All laptops must have full-disk encryption enabled and a screen "
        "lock timeout of 5 minutes or less. Lost or stolen devices must "
        "be reported to IT security within 1 hour of discovery.\n\n"
        "Passwords must be at least 14 characters and are rotated every "
        "180 days. Multi-factor authentication is required for all "
        "systems handling customer data.\n\n"
        "Third-party vendors requesting data access must complete a "
        "security review before any credentials are issued. Reviews are "
        "valid for 12 months."
    ),
}

# ---------------------------------------------------------------------------
# EMBEDDING — identical mechanic to ../embedding/embedding_search.py
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "what", "when", "where", "why",
    "each", "all", "must",
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
# STAGE 1: coarse retrieval over WHOLE documents
# ---------------------------------------------------------------------------
# Embedding a whole document as a single vector is cheap (one embedding
# call per document, however long) but imprecise — it averages together
# every topic the document covers. That's fine for THIS stage's job:
# roughly "is this document even in the right ballpark", not "which
# sentence answers the question."
COARSE_INDEX = {doc_id: embed(text) for doc_id, text in DOCUMENTS.items()}


def coarse_search(query: str, top_n_docs: int = 1) -> list[tuple[str, float]]:
    query_vector = embed(query)
    scored = [(doc_id, cosine_similarity(query_vector, vec)) for doc_id, vec in COARSE_INDEX.items()]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n_docs]


# ---------------------------------------------------------------------------
# STAGE 2: fine retrieval within the shortlisted documents only
# ---------------------------------------------------------------------------
def paragraph_chunks(doc_id: str) -> list[str]:
    """Split one document into paragraph-level chunks (on blank lines)."""
    return [p.strip() for p in DOCUMENTS[doc_id].split("\n\n") if p.strip()]


def fine_search(query: str, candidate_doc_ids: list[str], top_k_chunks: int = 2) -> list[tuple[str, str, float]]:
    """Chunk and embed ONLY the shortlisted documents — this is the
    expensive, precise pass, deliberately scoped down to a small subset
    instead of running over the whole corpus.
    """
    query_vector = embed(query)
    scored = []
    for doc_id in candidate_doc_ids:
        for chunk in paragraph_chunks(doc_id):
            score = cosine_similarity(query_vector, embed(chunk))
            scored.append((doc_id, chunk, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k_chunks]


def answer_query(query: str, top_n_docs: int = 1, top_k_chunks: int = 2) -> str:
    coarse_results = coarse_search(query, top_n_docs=top_n_docs)
    print(f"  [stage 1 - coarse: shortlisted {len(coarse_results)} of {len(DOCUMENTS)} documents]")
    for doc_id, score in coarse_results:
        print(f"    {doc_id}: {score:.3f}")

    candidate_doc_ids = [doc_id for doc_id, _ in coarse_results]
    fine_results = fine_search(query, candidate_doc_ids, top_k_chunks=top_k_chunks)
    total_candidate_chunks = sum(len(paragraph_chunks(d)) for d in candidate_doc_ids)
    print(f"  [stage 2 - fine: ranked {total_candidate_chunks} candidate sections, kept top {len(fine_results)}]")
    for doc_id, chunk, score in fine_results:
        print(f"    [{doc_id}] (similarity: {score:.3f}) {chunk[:70]}...")

    context = "\n\n".join(f"[Source: {doc_id}]\n{chunk}" for doc_id, chunk, _ in fine_results)
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

    total_paragraphs = sum(len(paragraph_chunks(d)) for d in DOCUMENTS)
    print(f"Company docs assistant — {len(DOCUMENTS)} documents, {total_paragraphs} sections total.")
    print("Coarse-to-fine retrieval demo. Type 'exit' to end the conversation.\n")

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
