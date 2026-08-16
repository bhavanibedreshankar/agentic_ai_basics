# corrective_rag

Corrective RAG (CRAG) and Self-RAG — making retrieval self-correcting instead of blindly trusting whatever came back, by grading retrieval quality before generation and checking groundedness after it.

## corrective_rag.py

An internal engineering-wiki assistant (deploy process, on-call rotation, incident severity, DB migration policy) that grades every retrieval as correct/ambiguous/incorrect before generating, falls back to Claude's real `web_search` tool when local retrieval is graded incorrect, and runs a groundedness check on the finished answer. Type `exit` to end the conversation.

### Concepts covered

- **`grade_retrieval(query, chunks)`** — the CRAG grading step: a structured-output judge call (same shape as `../../08_Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py`'s `evaluate_output`) scoring retrieval as `correct`/`ambiguous`/`incorrect`, run BEFORE generation — unlike `../rag/basic_rag.py`, which injects whatever top-k chunks it finds with no quality check at all.
- **`web_search_fallback(query)` / `corrective_retrieve(query)`** — the CORRECTION: on an `incorrect` grade, local retrieval is discarded and Claude's real, live `web_search` tool (same declaration as `../../03_Tools_and_Actions/web_search/web_search.py`) is called instead; on `ambiguous`, both sources are combined rather than picking one.
- **`refine_chunks(query, chunks)`** — CRAG's "decompose-then-recompose" step: even a `correct` retrieval gets filtered down to only the individual sentences that score well against the query, reusing the same `embed()`/`cosine_similarity()` mechanic at sentence granularity instead of chunk granularity.
- **`reflect_groundedness(answer, context)`** — the Self-RAG-style piece, run AFTER generation: checks whether the finished answer is actually supported by its context, approximating Self-RAG's ISSUP reflection token as an explicit judge call rather than a trained-in one (the module docstring explains exactly why a faithful reproduction isn't possible via prompting alone).
- **`answer_query(query)`** — wires grade → correct/fallback/combine → generate → reflect into one pipeline, printing each stage's verdict so the correction is visible, not just the final answer.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 09_RAG_and_Knowledge/corrective_rag/corrective_rag.py
```

Try:

```
You: What database migrations require review from the data platform team before merging?
  [grade: correct] The passage directly states migrations require data platform team review.
  [context source: local knowledge base (refined)]
  [self-reflection: grounded=True] Every claim traces back to the db-migration-policy passage.

Claude: Schema migrations must be reviewed by a member of the data platform team...

You: What year was the CAP theorem first published?
  [grade: incorrect] None of the passages mention the CAP theorem.
  [correction: local retrieval insufficient, falling back to live web search]
  [context source: web search (local retrieval discarded)]
  ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `JUDGE_MAX_TOKENS`, `EFFORT` — see `../../01_Core_Architecture/basics/README.md`
- `DOCUMENTS` — the internal knowledge base; add an entry to try grading against different content
- `refine_chunks`'s `min_score` — the sentence-level relevance cutoff for knowledge refinement

### See also

- `../rag/README.md` — the ungraded baseline this template adds correction on top of
- `../../03_Tools_and_Actions/web_search/README.md` — the same server-side `web_search` tool declaration, used here as CRAG's correction fallback instead of a standalone research tool
- `../../08_Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — the reusable single-call scoring shape both `grade_retrieval` and `reflect_groundedness` are built from
- `../../13_Benchmarking/trace_evaluation/README.md` — a different self-checking idea: scoring an already-finished trace, rather than gating a pipeline's own stages in real time
