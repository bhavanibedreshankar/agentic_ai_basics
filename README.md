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
| [`token_tracking/`](token_tracking/README.md) | Measuring and monitoring token usage and cost — reading `usage`, pre-flight counting, cumulative tracking | `basic_token_tracking.py` |
| [`prompt_chaining/`](prompt_chaining/README.md) | Decomposing a task into a fixed sequence of focused LLM calls, with a programmatic gate between steps | `basic_prompt_chaining.py` |
| [`context_management/`](context_management/README.md) | Keeping the context window relevant as a conversation grows: pruning, summarization, and retrieval | `pruning.py`, `summarization.py`, `retrieval.py` |
| [`RAG_and_Knowledge/`](RAG_and_Knowledge/README.md) | Retrieval-Augmented Generation and its building blocks: embedding, chunking, hybrid dense+sparse search, and coarse-to-fine retrieval | `embedding_search.py`, `chunking_strategies.py`, `basic_rag.py`, `hybrid_search.py`, `coarse_to_fine_retrieval.py` |
| [`Planning_and_Reasoning/`](Planning_and_Reasoning/README.md) | Getting Claude to reason more reliably: ReAct, chain-of-thought, tree of thought, plan-and-execute, and self-reflection | `react_agent.py`, `chain_of_thought.py`, `tree_of_thought.py`, `plan_and_execute.py`, `self_reflection.py` |
| [`Memory/`](Memory/README.md) | Six kinds of agent memory: in-context, working, episodic, basic long-term facts, structured semantic, and external/vector store | `in_context_memory.py`, `working_memory.py`, `episodic_memory.py`, `basic_agentic_memory.py`, `semantic_memory.py`, `external_memory.py` |
| [`Tools_and_Actions/`](Tools_and_Actions/README.md) | Tools that affect the world beyond text: tool-calling basics, a scalable tool registry, code execution, web search, file editing, external API/MCP connectors, and browser automation | `basic_agentic_tools.py`, `basic_tool_registry.py`, `code_interpreter.py`, `web_search.py`, `file_io_tools.py`, `api_connectors_mcp.py`, `browser_computer_use.py` |
| [`Multi_Agent_Systems/`](Multi_Agent_Systems/README.md) | Multiple agents working together: an orchestrator delegating to specialists, a reusable worker agent, permanent handoff, a validating supervisor, and a parallel uncoordinated swarm | `orchestrator.py`, `worker_agent.py`, `agent_handoff.py`, `supervisor_pattern.py`, `swarm.py` |

Each directory's README explains the concept, what the code demonstrates, how to run it, example output, and the config knobs you can tune.
