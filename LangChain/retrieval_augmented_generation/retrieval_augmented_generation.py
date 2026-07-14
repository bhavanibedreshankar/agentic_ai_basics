"""
CONCEPT: Retrieval-Augmented Generation — grounding a model's answer in
retrieved documents by making retrieval one more step in a pipeline,
rather than something the model has to ask for.

This template is the LangChain-native version of two existing templates:
  - ../../RAG_and_Knowledge/embedding/embedding_search.py hand-rolls a
    dependency-free `embed()` (word-hash buckets) and exposes retrieval as
    a TOOL Claude decides to call mid-conversation.
  - ../../RAG_and_Knowledge/rag/basic_rag.py wires retrieval + generation
    together but still through the tool-calling loop.
  Here, retrieval is just a `Runnable` (a `VectorStoreRetriever`) composed
  directly into an LCEL chain with `|`, the same way ../chains/chains.py
  composes prompt -> model -> parser steps. The model never decides
  WHETHER to retrieve — the chain always retrieves, then always answers
  from what it found. That's a real behavioral difference, not just a
  styling one: a tool-based RAG agent can choose to skip search and answer
  from general knowledge; this chain-based pipeline cannot.

The embedding function (`HashEmbeddings` below) reuses the exact same
word-hash-bucket technique as embedding_search.py's `embed()` — see that
file's docstring for the honest limitations (word overlap, not true
meaning). What's new here is `Embeddings` and `InMemoryVectorStore`:
LangChain's standard interfaces for "turn text into vectors" and "store
and search vectors", which any real embedding model or vector database
(Voyage AI, Pinecone, Chroma, ...) also implements — swap `HashEmbeddings`
for a real one and every other line in this file keeps working unchanged,
same promise embedding_search.py makes about swapping out `embed()`.

Use case: the same small documentation knowledge base as
embedding_search.py, answered through an LCEL retrieve-then-generate
chain instead of a tool call. Type 'exit' to end the conversation.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

EMBEDDING_DIM = 64
_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "with", "for", "on", "in",
    "at", "by", "as", "from", "how", "do", "does", "did", "i", "you",
    "your", "my", "me", "we", "they", "he", "she", "what", "when", "where",
    "why", "which", "let", "lets", "can", "will", "use", "used", "using",
}


class HashEmbeddings(Embeddings):
    """CONCEPT: implementing LangChain's `Embeddings` interface — just two
    methods, `embed_documents` (many texts at once, for indexing) and
    `embed_query` (one text, for a search). Any class with these two
    methods can be dropped into an `InMemoryVectorStore` (or any other
    LangChain vector store) in place of a real embeddings API client. See
    ../../RAG_and_Knowledge/embedding/embedding_search.py's `embed()` docstring for why this
    particular hashing scheme is a stand-in for a real model, not a real
    one itself.
    """

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for raw_word in text.lower().split():
            word = raw_word.strip(".,!?()`:;\"'")
            if not word or word in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
            vector[bucket] += 1.0
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude == 0:
            return vector
        return [v / magnitude for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


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

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the provided context. If the "
            "context doesn't cover it, say so honestly instead of guessing.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def build_retriever() -> VectorStoreRetriever:
    # CONCEPT: building the index once, up front — same principle as
    # embedding_search.py's module-level INDEX, just built through
    # InMemoryVectorStore.from_texts instead of a hand-rolled list of
    # (doc_id, text, vector) tuples.
    doc_ids = list(KNOWLEDGE_BASE.keys())
    texts = [KNOWLEDGE_BASE[doc_id] for doc_id in doc_ids]
    metadatas = [{"doc_id": doc_id} for doc_id in doc_ids]
    store = InMemoryVectorStore.from_texts(texts, HashEmbeddings(), metadatas=metadatas)
    return store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{doc.metadata['doc_id']}] {doc.page_content}" for doc in docs)


def build_rag_chain(llm: BaseChatModel, retriever: VectorStoreRetriever) -> Runnable:
    """Compose retrieval + generation into one pipeline. Takes `llm` and
    `retriever` as parameters so tests can substitute fakes — see this
    directory's README for how that's verified without a real API key.
    """
    # CONCEPT: the retriever is just another Runnable — `retriever |
    # format_docs` turns a question straight into a formatted context
    # string, and that whole sub-pipeline slots into the "context" key
    # alongside RunnablePassthrough() carrying the original question
    # through to "question" — the same fan-out-then-recombine shape
    # ../chains/chains.py's RunnableParallel uses for summary/sentiment.
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    retriever = build_retriever()
    chain = build_rag_chain(llm, retriever)

    print(f"Documentation assistant — {len(KNOWLEDGE_BASE)} documents indexed (LCEL RAG demo).")
    print("Type 'exit' to end the conversation.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        docs = retriever.invoke(question)
        print(f"  [retrieved: {[d.metadata['doc_id'] for d in docs]}]")

        answer = chain.invoke(question)
        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()
