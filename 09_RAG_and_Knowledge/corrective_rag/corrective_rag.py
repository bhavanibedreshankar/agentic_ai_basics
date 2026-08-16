"""
CONCEPT: Corrective RAG (CRAG) and Self-RAG — two ways of making retrieval
SELF-CORRECTING instead of blindly trusting whatever came back, which is
exactly what ../rag/basic_rag.py does: retrieve top-k chunks, inject them,
generate, no questions asked. If the retriever returns three irrelevant
chunks, basic_rag.py generates an answer grounded in irrelevant chunks
anyway. Both patterns below exist to catch that before it reaches the user.

CORRECTIVE RAG (implemented here for real): a lightweight GRADE step runs
right after retrieval, BEFORE generation. A judge call scores the retrieved
chunks against the query as "correct" (good enough, use as-is/refined),
"ambiguous" (partially relevant, worth combining with another source), or
"incorrect" (not relevant at all — don't generate from this). On
"incorrect", the pipeline doesn't just shrug and generate anyway like
../rag/basic_rag.py would — it falls back to Claude's real, live
`web_search` tool (same server-side tool as
../../03_Tools_and_Actions/web_search/web_search.py) to find better
grounding instead. This is CRAG's actual design: retrieval quality is
graded, and a bad grade triggers a CORRECTION, not silent pass-through.

SELF-RAG (approximated here, honestly): the real Self-RAG technique bakes
reflection into the GENERATING model itself — a specially fine-tuned model
emits inline "reflection tokens" (ISREL/ISSUP/ISUSE) as part of decoding,
deciding whether to retrieve, whether each passage is relevant, and
whether its own output is actually supported by what it retrieved — all
inside one forward pass. That requires training a model to produce those
tokens; it is NOT something you can faithfully reproduce by prompting an
off-the-shelf model. What this file does instead is the practical
approximation used throughout this repo (see `evaluate_output` in
../../08_Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py):
a SEPARATE judge call, after generation, that checks whether the answer is
actually SUPPORTED by the context it was given (the ISSUP idea) — same
spirit as Self-RAG's self-reflection, implemented as an explicit extra
step rather than a trained-in one.

Put together: CRAG's grade-then-correct step runs BEFORE generation
(fixing bad retrieval), and the Self-RAG-style groundedness check runs
AFTER generation (catching an answer that drifted from its context even
when retrieval itself was fine) — two different failure modes, two
different places in the pipeline to catch them.

Use case: an internal engineering wiki (deploy process, on-call rotation,
incident severity, DB migration policy) — reusing the same embed()/
chunk_text() mechanic as ../rag/basic_rag.py, but with a grading step in
front of generation and a reflection step behind it. Type 'exit' to end
the conversation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys

import anthropic

# --- API settings (see ../../01_Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 300
EFFORT = "medium"

RAG_SYSTEM_PROMPT = (
    "You are an internal engineering assistant answering questions using "
    "ONLY the context provided below. If the context doesn't contain the "
    "answer, say so explicitly rather than guessing or using outside "
    "knowledge."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# A small, PRIVATE knowledge base — the kind of internal-only content a
# public web search could never answer, so the demo queries below can
# clearly separate "local retrieval is good enough" from "local retrieval
# has nothing, go find real grounding elsewhere."
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "deploy-process": (
        "Deploys to production require a passing CI run and one approving "
        "code review. Deploys are only allowed Monday through Thursday "
        "before 3pm local time, to keep someone available to respond if "
        "something breaks. Rollbacks can be triggered by any engineer via "
        "the deploy dashboard without needing a second approval."
    ),
    "oncall-rotation": (
        "The on-call rotation is weekly, handed off every Monday at 10am. "
        "The primary on-call engineer is paged first; if there's no "
        "acknowledgment within 10 minutes, the secondary is paged "
        "automatically. On-call engineers are expected to be reachable "
        "within 5 minutes during their rotation."
    ),
    "incident-severity": (
        "SEV1 incidents mean full service outage or data loss risk and "
        "require immediate all-hands response. SEV2 means significant "
        "degraded functionality for a large subset of users. SEV3 is a "
        "minor issue with a workaround available. Every SEV1 and SEV2 "
        "requires a written postmortem within 3 business days."
    ),
    "db-migration-policy": (
        "Schema migrations must be reviewed by a member of the data "
        "platform team before merging, in addition to normal code review. "
        "Migrations must be backward-compatible for at least one release "
        "cycle before old columns are dropped, so a mid-deploy rollback "
        "never lands on an incompatible schema."
    ),
}

# ---------------------------------------------------------------------------
# CHUNKING + EMBEDDING — identical mechanic to ../rag/basic_rag.py; see
# that file's comments for the full explanation of the hash-based
# stand-in for a real embeddings API.
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


INDEX: list[tuple[str, str, list[float]]] = []
for doc_id, doc_text in DOCUMENTS.items():
    for chunk in chunk_text(doc_text):
        INDEX.append((doc_id, chunk, embed(chunk)))


def retrieve(query: str, top_k: int = 2) -> list[tuple[str, str, float]]:
    query_vector = embed(query)
    scored = [(doc_id, chunk, cosine_similarity(query_vector, vec)) for doc_id, chunk, vec in INDEX]
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# CONCEPT: the CRAG grading step. A structured-output judge call, same
# shape as ../../08_Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py's
# evaluate_output — except it grades RETRIEVAL quality against a query,
# not a finished answer against a rubric, and runs BEFORE generation
# instead of after.
# ---------------------------------------------------------------------------
GRADE_SYSTEM_PROMPT = (
    "You grade whether retrieved passages are sufficient to answer a "
    "question. Respond with verdict 'correct' if the passages directly "
    "answer the question, 'ambiguous' if they're partially relevant but "
    "incomplete, or 'incorrect' if they don't address the question at all."
)

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "ambiguous", "incorrect"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}


def grade_retrieval(query: str, chunks: list[tuple[str, str, float]]) -> tuple[str, str]:
    passages = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text, _ in chunks)
    prompt = f"Question: {query}\n\nRetrieved passages:\n{passages}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        system=GRADE_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": GRADE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads("".join(b.text for b in response.content if b.type == "text"))
    return result["verdict"], result["reasoning"]


# ---------------------------------------------------------------------------
# CONCEPT: knowledge refinement — CRAG's "decompose-then-recompose" step.
# Even a "correct" retrieval can contain sentences that don't actually
# matter to the question; this keeps only the individual sentences
# ("strips") that score reasonably well against the query, instead of
# injecting the whole chunk verbatim. Reuses the same embed()/
# cosine_similarity() mechanic already built above, just applied at
# sentence granularity instead of chunk granularity.
# ---------------------------------------------------------------------------
def refine_chunks(query: str, chunks: list[tuple[str, str, float]], min_score: float = 0.15) -> str:
    query_vector = embed(query)
    kept: list[str] = []
    for _doc_id, text, _score in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if cosine_similarity(query_vector, embed(sentence)) >= min_score:
                kept.append(sentence)
    return " ".join(kept) if kept else " ".join(text for _, text, _ in chunks)


# ---------------------------------------------------------------------------
# CONCEPT: the CORRECTION — a REAL fallback to Claude's server-side
# `web_search` tool (same declaration as
# ../../03_Tools_and_Actions/web_search/web_search.py), used only when
# local retrieval was graded "incorrect". This is what makes it
# CORRECTIVE rather than just a quality filter: a bad grade doesn't just
# get flagged, it triggers going and finding better grounding.
# ---------------------------------------------------------------------------
def web_search_fallback(query: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system="Search the web and report back only the specific facts relevant to the question, concisely.",
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 2}],
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": query}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def corrective_retrieve(query: str) -> tuple[str, str, str]:
    """Returns (context, verdict, source_description) — the full CRAG
    grade-then-correct pipeline: retrieve -> grade -> branch.
    """
    chunks = retrieve(query)
    verdict, reasoning = grade_retrieval(query, chunks)
    print(f"  [grade: {verdict}] {reasoning}")

    if verdict == "correct":
        return refine_chunks(query, chunks), verdict, "local knowledge base (refined)"
    if verdict == "incorrect":
        print("  [correction: local retrieval insufficient, falling back to live web search]")
        return web_search_fallback(query), verdict, "web search (local retrieval discarded)"
    # ambiguous: combine both sources rather than picking one
    print("  [correction: retrieval ambiguous, combining local + web search]")
    local_context = refine_chunks(query, chunks)
    web_context = web_search_fallback(query)
    combined = f"From internal docs:\n{local_context}\n\nFrom web search:\n{web_context}"
    return combined, verdict, "local knowledge base + web search (combined)"


# ---------------------------------------------------------------------------
# CONCEPT: the Self-RAG-style reflection step — ISSUP, approximated. A
# second judge call checking whether the ANSWER is actually supported by
# the CONTEXT it was generated from, run after generation completes. This
# is a genuinely different failure mode than a bad CRAG grade: retrieval
# can be perfectly relevant and the model can still write something the
# context doesn't actually support (an added detail, an overgeneralization).
# Honest simplification: real Self-RAG would let the model re-retrieve or
# regenerate on a failed check, inline, as part of decoding. Here the
# result is just surfaced to the user — looping back into
# corrective_retrieve() on failure is a natural extension left as a
# comment, not implemented, to keep this demo's control flow readable.
# ---------------------------------------------------------------------------
REFLECT_SYSTEM_PROMPT = (
    "You check whether an answer is fully supported by its given context. "
    "Respond with grounded=true only if every claim in the answer is "
    "directly backed by the context, with no invented or outside details."
)

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["grounded", "feedback"],
    "additionalProperties": False,
}


def reflect_groundedness(answer: str, context: str) -> tuple[bool, str]:
    prompt = f"Context:\n{context}\n\nAnswer:\n{answer}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        system=REFLECT_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": REFLECT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads("".join(b.text for b in response.content if b.type == "text"))
    return result["grounded"], result["feedback"]


def answer_query(query: str) -> str:
    context, _verdict, source = corrective_retrieve(query)
    print(f"  [context source: {source}]")

    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=RAG_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")

    grounded, feedback = reflect_groundedness(answer, context)
    print(f"  [self-reflection: grounded={grounded}] {feedback}")

    return answer


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Corrective RAG demo — internal engineering wiki, with retrieval grading and web-search correction.")
    print("Type 'exit' to end the conversation.\n")
    print('Try: "What database migrations require review from the data platform team before merging?" (correct)')
    print('Try: "Who gets paged for a SEV1 during a deploy rollback?" (ambiguous)')
    print('Try: "What year was the CAP theorem first published?" (incorrect -> web fallback)\n')

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
