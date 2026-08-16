"""
CONCEPT: Chunking — splitting a document into smaller pieces before it gets
embedded and indexed, so retrieval can return the specific relevant
section instead of an entire (possibly huge) document.

Why chunk at all? Two reasons:
  1. Embedding models (and Claude's context window) have limits on how
     much text fits in one call — a 50-page document can't be embedded
     as a single vector meaningfully; too much gets averaged away.
  2. Precision — if a document covers ten topics and a query is about
     one of them, returning the whole document wastes context on nine
     irrelevant topics. A well-sized chunk returns just the relevant
     part.

This template is pure text processing — no embeddings, no API calls, no
`ANTHROPIC_API_KEY` needed. Chunking always happens locally, BEFORE any
embedding step (see ../embedding/embedding_search.py and ../rag/basic_rag.py,
which both embed AFTER chunking, not instead of it).

There's no single "correct" chunk size — it's a trade-off:
  - Chunks too SMALL lose surrounding context (a sentence fragment about
    "it" with no antecedent nearby is useless on its own).
  - Chunks too LARGE dilute relevance (a chunk covering five topics scores
    only weakly for any one of them) and cost more tokens once retrieved.

Run this file directly to see all three strategies applied to the same
sample document, side by side:

    python3 chunking_strategies.py
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# A sample multi-paragraph document to chunk. Deliberately long enough that
# the different strategies produce visibly different results.
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT = """
Retrieval-augmented generation, or RAG, is a technique for grounding a
language model's answers in an external knowledge base rather than
relying only on what the model memorized during training. Instead of
asking the model to answer from memory, the system first retrieves
relevant documents and includes them as context in the prompt.

The core pipeline has three stages. First, documents are split into
chunks and converted into vector embeddings, then stored in an index.
Second, when a query arrives, it's embedded the same way and compared
against the index to find the most similar chunks. Third, those chunks
are injected into the prompt alongside the original query, and the model
generates an answer grounded in that retrieved context.

Chunk size is one of the most consequential design decisions in a RAG
system. Chunks that are too small lose context — a sentence pulled out of
its paragraph can be ambiguous or meaningless on its own. Chunks that are
too large dilute relevance — a chunk covering several unrelated topics
will only weakly match a query about any single one of them, and wastes
tokens once retrieved.

Overlap between consecutive chunks helps mitigate the boundary problem:
without it, a sentence that happens to fall right at a chunk boundary can
be split in half, with neither resulting chunk containing the complete
thought. A modest overlap, often ten to twenty percent of the chunk size,
ensures that content near a boundary appears intact in at least one
chunk.
""".strip()


# ---------------------------------------------------------------------------
# STRATEGY 1: fixed-size chunking
# ---------------------------------------------------------------------------
def chunk_fixed_size(text: str, chunk_size: int = 300) -> list[str]:
    """CONCEPT: the simplest possible strategy — cut the text every
    `chunk_size` characters, with no regard for word or sentence
    boundaries. Fast and predictable, but can (and will) slice a word or
    sentence in half right at the cut point.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ---------------------------------------------------------------------------
# STRATEGY 2: sentence-based chunking
# ---------------------------------------------------------------------------
def chunk_by_sentences(text: str, max_chunk_size: int = 300) -> list[str]:
    """CONCEPT: respect natural language boundaries. Split into sentences
    first, then greedily pack whole sentences into a chunk until adding
    the next one would exceed max_chunk_size. Never cuts a sentence in
    half — the trade-off is that chunk sizes vary rather than being
    exactly uniform.
    """
    # A simple sentence splitter: break after '.', '!', or '?' followed by
    # whitespace. Not linguistically perfect (doesn't special-case "Dr."
    # or "e.g."), but good enough for this demonstration.
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())

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
# STRATEGY 3: fixed-size chunking WITH overlap
# ---------------------------------------------------------------------------
def chunk_with_overlap(text: str, chunk_size: int = 300, overlap: int = 60) -> list[str]:
    """CONCEPT: mitigate the boundary problem. Same fixed-size approach as
    chunk_fixed_size, but each chunk repeats the last `overlap` characters
    of the previous one. If something important sits right at a boundary,
    it now appears complete in at least one chunk instead of being split
    across two incomplete halves.

    The cost: more total characters get embedded (each chunk after the
    first re-embeds `overlap` characters it already covered), which means
    more embedding calls' worth of cost in a real system — overlap isn't
    free, it's a deliberate trade of some redundancy for better recall.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def describe(name: str, chunks: list[str]) -> None:
    sizes = [len(c) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    print(f"\n=== {name} ===")
    print(f"{len(chunks)} chunks, average {avg_size:.0f} characters each")
    for i, chunk in enumerate(chunks):
        preview = chunk.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  [{i}] ({len(chunk)} chars) {preview}")


def main() -> None:
    print(f"Sample document: {len(SAMPLE_DOCUMENT)} characters\n")
    print("Comparing chunking strategies on the same document...")

    describe("Fixed-size (300 chars, no overlap)", chunk_fixed_size(SAMPLE_DOCUMENT, chunk_size=300))
    describe("Sentence-based (max 300 chars)", chunk_by_sentences(SAMPLE_DOCUMENT, max_chunk_size=300))
    describe("Fixed-size with overlap (300 chars, 60 overlap)", chunk_with_overlap(SAMPLE_DOCUMENT, chunk_size=300, overlap=60))

    print(
        "\nNotice: fixed-size chunking can cut mid-word or mid-sentence at "
        "the boundary; sentence-based chunking never does, at the cost of "
        "uneven chunk sizes; overlap increases the total character count "
        "embedded (see ../rag/basic_rag.py and ../hybrid_rag/hybrid_search.py "
        "for what happens to these chunks after this step — they get "
        "embedded and indexed for retrieval)."
    )


if __name__ == "__main__":
    main()
