# Agentic AI Basics

A learning series of small, self-contained Python templates for building with the Claude API — each one focused on a single agentic AI concept, with comments explaining the "why" as much as the "what". Every template lives in its own directory with a dedicated README going into that concept in depth; this file just covers setup and a quick index of everything here.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

Don't have a key? Create one at [platform.claude.com](https://platform.claude.com).

Each template is run from the repo root, e.g.:

```bash
python3 basics/basic.py
```

Most templates are interactive chat loops — type `exit` to end the conversation.

## Templates

| Directory | Concept | File(s) |
|---|---|---|
| [`basics/`](basics/README.md) | A single request/response call — API settings, auth, and the core `ask_claude` pattern everything else builds on | `basic.py` |
| [`agentic_loop/`](agentic_loop/README.md) | The multi-turn conversation loop — how a "chat" is built on top of a stateless API | `basic_agentic_loop.py` |
| [`tool_use/`](tool_use/README.md) | Tool use (function calling) — JSON Schema tool definitions, parallel calls, error handling, the agentic tool-calling loop | `basic_agentic_tools.py` |
| [`memory_management/`](memory_management/README.md) | Short-term vs. long-term memory — persisting facts across sessions, context injection, sliding-window trimming | `basic_agentic_memory.py` |
| [`token_tracking/`](token_tracking/README.md) | Measuring and monitoring token usage and cost — reading `usage`, pre-flight counting, cumulative tracking | `basic_token_tracking.py` |
| [`prompt_chaining/`](prompt_chaining/README.md) | Decomposing a task into a fixed sequence of focused LLM calls, with a programmatic gate between steps | `basic_prompt_chaining.py` |
| [`tool_registry/`](tool_registry/README.md) | A catalog-driven pattern for managing many tools — one registry drives both tool definitions and dispatch | `basic_tool_registry.py` |
| [`context_management/`](context_management/README.md) | Keeping the context window relevant as a conversation grows: pruning, summarization, and retrieval | `pruning.py`, `summarization.py`, `retrieval.py` |
| [`RAG_and_Knowledge/`](RAG_and_Knowledge/README.md) | Retrieval-Augmented Generation and its building blocks: embedding, chunking, hybrid dense+sparse search, and coarse-to-fine retrieval | `embedding_search.py`, `chunking_strategies.py`, `basic_rag.py`, `hybrid_search.py`, `coarse_to_fine_retrieval.py` |
| [`Planning_and_Reasoning/`](Planning_and_Reasoning/README.md) | Getting Claude to reason more reliably: ReAct, chain-of-thought, tree of thought, plan-and-execute, and self-reflection | `react_agent.py`, `chain_of_thought.py`, `tree_of_thought.py`, `plan_and_execute.py`, `self_reflection.py` |

Each directory's README explains the concept, what the code demonstrates, how to run it, example output, and the config knobs you can tune.
