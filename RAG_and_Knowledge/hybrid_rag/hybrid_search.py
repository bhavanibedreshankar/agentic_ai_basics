"""
CONCEPT: Hybrid RAG — combining dense vector retrieval (semantic
similarity via embeddings) with sparse keyword retrieval (BM25) so each
can cover the other's blind spot.

Dense (embedding) search is good at matching MEANING even when the exact
words differ — but ../embedding/embedding_search.py's docstring already
flags its weak spot: rare, specific terms (an error code, a product SKU,
an exact version number) can get diluted among common words instead of
standing out. Sparse keyword search (BM25) is the opposite: it's
EXCELLENT at exact term matching — the rarer a term is across the corpus,
the more heavily BM25 weights a document that contains it — but it has no
notion of meaning at all, so a query using different wording than the
source document scores poorly, exactly where dense search does well.

Combining both and merging their rankings — "hybrid" search — covers more
ground than either alone. This template builds both a dense index and a
BM25 index over the same documents, scores a query against each
independently, normalizes the two score scales so they're comparable, and
blends them into one ranking.

Run this file directly to see all three rankings (dense-only,
sparse-only, hybrid) for the same query, side by side, then ask Claude a
question grounded in the hybrid results:

    python3 hybrid_search.py
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
    "You are a support assistant. Answer using ONLY the context provided "
    "below. If the context doesn't cover the question, say so explicitly."
)

# ---------------------------------------------------------------------------
# A support knowledge base deliberately containing a RARE, EXACT term (an
# error code) that dense search tends to under-rank and BM25 excels at.
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "password-reset": "To reset your password, go to Settings > Security and click 'Reset Password'. A reset link is emailed to your account address and expires after 24 hours.",
    "error-e4521": "Error E-4521: Authentication token expired. This happens when a session has been idle too long. To resolve, clear your browser cache and log in again.",
    "billing-update": "To update your billing information, go to Settings > Billing > Payment Methods. Changes take effect on your next billing cycle.",
    "two-factor-setup": "Two-factor authentication can be enabled under Settings > Security > 2FA. Supported methods are authenticator apps and SMS codes.",
    "error-e1032": "Error E-1032: Payment declined by processor. This usually means the card was expired or the bank flagged the charge. Try a different payment method or contact your bank.",
}

# ---------------------------------------------------------------------------
# DENSE (embedding) index — identical mechanic to ../embedding/embedding_search.py
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "what", "when", "where", "why",
}


def tokenize(text: str) -> list[str]:
    words = []
    for raw_word in text.lower().split():
        word = raw_word.strip(".,!?()`:;\"'-")
        if word and word not in _STOPWORDS:
            words.append(word)
    return words


def embed(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for word in tokenize(text):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[bucket] += 1.0
    magnitude = math.sqrt(sum(v * v for v in vector))
    return vector if magnitude == 0 else [v / magnitude for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


DOC_IDS = list(DOCUMENTS.keys())
DENSE_INDEX = {doc_id: embed(text) for doc_id, text in DOCUMENTS.items()}


def dense_search(query: str) -> dict[str, float]:
    """Return every document's raw cosine similarity to the query."""
    query_vector = embed(query)
    return {doc_id: cosine_similarity(query_vector, vec) for doc_id, vec in DENSE_INDEX.items()}


# ---------------------------------------------------------------------------
# SPARSE (BM25) index
# ---------------------------------------------------------------------------
# CONCEPT: BM25 scores a document for a query by summing, for each query
# term, how much that term stands out in THIS document relative to the
# whole corpus. Two components combine into that score:
#   - Term frequency (TF): how often the term appears in this document,
#     with diminishing returns — the 5th occurrence matters much less
#     than the 1st (controlled by k1).
#   - Inverse document frequency (IDF): how RARE the term is across all
#     documents — a term that appears in every document (like "error")
#     tells you nothing distinctive; a term that appears in only one
#     document (like "e4521") is a near-perfect signal for that document.
#   - A length normalization factor (controlled by b) prevents longer
#     documents from scoring higher purely by containing more words.
K1 = 1.5  # controls how quickly repeated term occurrences stop adding score
B = 0.75  # controls how strongly document length is penalized

_DOC_TOKENS = {doc_id: tokenize(text) for doc_id, text in DOCUMENTS.items()}
_DOC_LENGTHS = {doc_id: len(tokens) for doc_id, tokens in _DOC_TOKENS.items()}
_AVG_DOC_LENGTH = sum(_DOC_LENGTHS.values()) / len(_DOC_LENGTHS)


def _idf(term: str) -> float:
    """How many of the N documents contain `term` at least once — fewer
    documents containing it means a higher (more distinctive) score.
    """
    n_docs = len(DOCUMENTS)
    n_containing = sum(1 for tokens in _DOC_TOKENS.values() if term in tokens)
    return math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1)


def _bm25_score(query_terms: list[str], doc_id: str) -> float:
    tokens = _DOC_TOKENS[doc_id]
    doc_length = _DOC_LENGTHS[doc_id]
    score = 0.0
    for term in query_terms:
        term_freq = tokens.count(term)
        if term_freq == 0:
            continue
        numerator = term_freq * (K1 + 1)
        denominator = term_freq + K1 * (1 - B + B * doc_length / _AVG_DOC_LENGTH)
        score += _idf(term) * (numerator / denominator)
    return score


def sparse_search(query: str) -> dict[str, float]:
    """Return every document's raw BM25 score for the query."""
    query_terms = tokenize(query)
    return {doc_id: _bm25_score(query_terms, doc_id) for doc_id in DOC_IDS}


# ---------------------------------------------------------------------------
# HYBRID: normalize both score sets to [0, 1], then blend
# ---------------------------------------------------------------------------
def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """CONCEPT: dense and sparse scores live on completely different
    scales — cosine similarity is bounded to [0, 1] here, but BM25 is
    unbounded and depends on corpus size and term rarity. Blending raw
    scores would let whichever metric happens to produce bigger numbers
    dominate. Min-max normalizing both to [0, 1] first makes them
    comparable before combining.
    """
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in scores.items()}


def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 3) -> list[tuple[str, float]]:
    """Blend normalized dense and sparse scores.

    `alpha` controls the mix: 1.0 is pure dense (semantic), 0.0 is pure
    sparse (keyword), 0.5 weighs them equally. Tune it based on how much
    your queries rely on exact terminology versus paraphrased meaning.
    """
    dense = _normalize(dense_search(query))
    sparse = _normalize(sparse_search(query))

    combined = {doc_id: alpha * dense[doc_id] + (1 - alpha) * sparse[doc_id] for doc_id in DOC_IDS}
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _print_ranking(label: str, scores: dict[str, float]) -> None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(f"  {label}: " + ", ".join(f"{doc_id} ({score:.3f})" for doc_id, score in ranked[:3]))


def answer_query(query: str) -> str:
    top_docs = hybrid_search(query)
    context = "\n\n".join(f"[Source: {doc_id}]\n{DOCUMENTS[doc_id]}" for doc_id, _ in top_docs)
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

    print(f"Support assistant — {len(DOCUMENTS)} documents indexed (hybrid dense + sparse search).")
    print("Type 'exit' to end the conversation.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue

        # Show all three rankings side by side before generating an
        # answer, so the effect of combining dense + sparse is visible.
        print("\n  --- ranking comparison ---")
        _print_ranking("dense only  ", _normalize(dense_search(query)))
        _print_ranking("sparse only ", _normalize(sparse_search(query)))
        _print_ranking("hybrid      ", dict(hybrid_search(query, top_k=len(DOCUMENTS))))
        print()

        answer = answer_query(query)
        print(f"Claude: {answer}\n")


if __name__ == "__main__":
    main()
