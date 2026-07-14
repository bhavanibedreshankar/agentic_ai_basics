# chains

Composing sequential/parallel LLM and tool calls into a single pipeline (LCEL).

## chains.py

A product review pipeline: summary and sentiment are computed in parallel from the same review text, then a branch either drafts a reply or escalates to a human agent depending on the sentiment. Type `exit` to end the session.

### Concepts covered

- **`RunnableParallel`** — runs the `summary` and `sentiment` sub-chains concurrently over the same input, since neither depends on the other's output.
- **`RunnablePassthrough`** — carries the original `review_text` through the parallel step unchanged, so it's still available to whichever branch runs next.
- **`RunnableBranch`** — a sequence of `(condition, runnable)` pairs plus a default, evaluated inline as part of the pipeline; the LCEL-native equivalent of a manual `if/else` gate.
- **`StrOutputParser`** — extracts plain text from the model's response, replacing the `"".join(b.text for b in response.content if ...)` pattern every raw-SDK template in this repo repeats by hand.
- **`build_review_chain`** — the whole pipeline as one composed `Runnable`, invoked with a single `.invoke({"review_text": ...})`.
- Contrast with [`../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py`](../../Agent_Frameworks_and_Patterns/prompt_chaining/README.md), which builds the same kind of fixed-sequence-with-a-gate pipeline by hand, strictly sequentially, with no parallel branch.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/chains/chains.py
```

Try:

```
Review: The battery life is terrible and it stopped charging after a week.

[escalated to a human agent] summary: Battery drains fast and stopped charging within a week. (sentiment: negative)
```

```
Review: Great battery life and it charges so fast!

Thanks so much for the kind words about the battery — glad it's working great for you!
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `SUMMARY_PROMPT` / `SENTIMENT_PROMPT` / `REPLY_PROMPT` — edit independently; each is its own small `ChatPromptTemplate`

### See also

- [`../prompt_templates/README.md`](../prompt_templates/README.md) — the template-building block chains compose
- [`../langgraph_workflows/README.md`](../langgraph_workflows/README.md) — branching as graph EDGES instead of a Runnable, including cycles this file's DAG-shaped pipeline can't express
