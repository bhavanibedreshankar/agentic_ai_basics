# retrieval_augmented_generation

Grounding responses in a vector store of external documents.

## retrieval_augmented_generation.py

The same small documentation knowledge base as [`../../RAG_and_Knowledge/embedding/embedding_search.py`](../../RAG_and_Knowledge/embedding/README.md), answered through an LCEL retrieve-then-generate chain instead of a tool call. Type `exit` to end the conversation.

### Concepts covered

- **`Embeddings`** — LangChain's two-method interface (`embed_documents`, `embed_query`) for turning text into vectors; `HashEmbeddings` implements it with the same dependency-free word-hash-bucket technique as `embedding_search.py`'s `embed()`.
- **`InMemoryVectorStore.from_texts`** — builds a searchable vector index once, up front, the same principle as `embedding_search.py`'s module-level `INDEX` but through a standard store interface any real vector database also implements.
- **`.as_retriever()`** — turns the vector store into a `Runnable` (`VectorStoreRetriever`) that composes directly into an LCEL chain with `|`.
- **`build_rag_chain`** — `{"context": retriever | format_docs, "question": RunnablePassthrough()} | RAG_PROMPT | llm | StrOutputParser()`, the same fan-out-then-recombine shape as [`../chains/chains.py`](../chains/README.md)'s `RunnableParallel`.
- Contrast with `embedding_search.py`, where retrieval is a tool the model can choose to call or skip; here retrieval always runs — the chain has no path that answers without it.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/retrieval_augmented_generation/retrieval_augmented_generation.py
```

Try:

```
You: How do branches work in git?
  [retrieved: ['git-basics', 'rest-apis']]

Claude: Branches let you develop features in isolation before merging them back into the main line of history.
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `EMBEDDING_DIM`, `search_kwargs={"k": 2}` in `build_retriever` — tune vector size and how many documents are retrieved per query

### See also

- [`../../RAG_and_Knowledge/embedding/README.md`](../../RAG_and_Knowledge/embedding/README.md) — the same `embed()` technique, exposed as a tool instead of a chain step
- [`../chains/README.md`](../chains/README.md) — the `RunnableParallel` fan-out pattern this chain reuses
