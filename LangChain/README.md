# LangChain

Eight building blocks of the LangChain framework, layered so each one leans on the last: a reusable prompt becomes a step in a pipeline, a pipeline gains memory and tools, tools get grounded in retrieved documents and validated output, and the whole thing gets both a graph-shaped alternative to linear pipelines and an outside-in way to observe any of it running. Every template here is built on `langchain-core` / `langchain` / `langchain-anthropic` (LangChain 1.x), with Claude as the model — contrast with the rest of this repo, which calls the `anthropic` SDK directly with no framework in between.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`prompt_templates/`](prompt_templates/README.md) | The base unit — a reusable, validated, parameterized prompt (`ChatPromptTemplate`, `.partial()`) |
| 2 | [`chains/`](chains/README.md) | Composing templates and models into pipelines with LCEL (`RunnableParallel`, `RunnableBranch`, `|`) |
| 3 | [`memory/`](memory/README.md) | Giving a chain per-session conversational state (`RunnableWithMessageHistory`) |
| 4 | [`agents_and_tools/`](agents_and_tools/README.md) | Letting the MODEL decide what happens next, not fixed pipeline code (`create_agent`, `@tool`) |
| 5 | [`retrieval_augmented_generation/`](retrieval_augmented_generation/README.md) | Grounding a chain's answer in a vector store (`Embeddings`, `InMemoryVectorStore`, `.as_retriever()`) |
| 6 | [`output_parsers/`](output_parsers/README.md) | Turning a model's free-form answer into a validated object (`PydanticOutputParser`, `with_structured_output`) |
| 7 | [`langgraph_workflows/`](langgraph_workflows/README.md) | A graph of nodes and conditional edges, when a linear pipeline isn't enough (`StateGraph`) — a taste of the full `../LangGraph/` topic |
| 8 | [`callbacks_and_tracing/`](callbacks_and_tracing/README.md) | Observing any chain from outside its own code (`BaseCallbackHandler`, `config={"callbacks": [...]}`) |

## Setup

Same as the rest of the repo, plus the LangChain packages:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
pip install langchain langchain-core langchain-anthropic langgraph pydantic
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 LangChain/prompt_templates/prompt_templates.py
```

## How these relate to each other

| | Adds | Still stateless/fixed? | Built on |
|---|---|---|---|
| `prompt_templates/` | Validated, reusable prompt text | Yes — one call, no history, no branching | — |
| `chains/` | Multi-step composition, parallelism, branching | Yes — shape fixed at compose time | `prompt_templates/` |
| `memory/` | Cross-turn conversational state | No — history persists across calls | `chains/`-style composition |
| `agents_and_tools/` | Model-decided control flow | No — number/order of steps unknown ahead of time | `prompt_templates/` (system prompt) + tools |
| `retrieval_augmented_generation/` | External knowledge grounding | Yes — retrieval always runs, chain shape fixed | `chains/`'s parallel-fan-out pattern |
| `output_parsers/` | Guaranteed output shape | Yes — same fixed pipeline shape, different last step | `chains/`'s "swap the last step" idea |
| `langgraph_workflows/` | Cycles + runtime-computed edges | No — graph shape can loop, unlike a DAG-shaped chain | `StateGraph`, replacing `|` composition |
| `callbacks_and_tracing/` | Cross-cutting observability | N/A — orthogonal to all the above; attaches to any of them | `BaseCallbackHandler` + any chain |

`prompt_templates/` and `chains/` are the two foundational LCEL building blocks everything else composes with. `memory/` and `agents_and_tools/` each remove one of chains/'s fixed properties — statelessness and fixed control flow, respectively. `retrieval_augmented_generation/` and `output_parsers/` stay chain-shaped but change what flows in (external documents) and out (validated objects). `langgraph_workflows/` is the odd one out — it replaces LCEL's `|` composition with an explicit graph, needed once branching has to be able to loop rather than just fork. `callbacks_and_tracing/` doesn't compose with the others at all; it observes whichever one you attach it to.
