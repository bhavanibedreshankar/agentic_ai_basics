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
python3 Core_Architecture/basics/basic.py
```

Most templates are interactive chat loops — type `exit` to end the conversation.

## Templates

| Directory | Concept | File(s) |
|---|---|---|
| [`Core_Architecture/`](Core_Architecture/README.md) | What an agent is actually made of: a single request/response call, token usage tracking, the reasoning-engine LLM, tool use/function calling, and the system prompt that shapes it all | `basic.py`, `basic_token_tracking.py`, `agent.py`, `llm_backbone.py`, `basic_agentic_tools.py`, `system_prompt.py` |
| [`RAG_and_Knowledge/`](RAG_and_Knowledge/README.md) | Retrieval-Augmented Generation and its building blocks: embedding, chunking, hybrid dense+sparse search, and coarse-to-fine retrieval | `embedding_search.py`, `chunking_strategies.py`, `basic_rag.py`, `hybrid_search.py`, `coarse_to_fine_retrieval.py` |
| [`Planning_and_Reasoning/`](Planning_and_Reasoning/README.md) | Getting Claude to reason more reliably: ReAct, chain-of-thought, tree of thought, plan-and-execute, and self-reflection | `react_agent.py`, `chain_of_thought.py`, `tree_of_thought.py`, `plan_and_execute.py`, `self_reflection.py` |
| [`Memory/`](Memory/README.md) | Six kinds of agent memory: in-context, working, episodic, basic long-term facts, structured semantic, and external/vector store | `in_context_memory.py`, `working_memory.py`, `episodic_memory.py`, `basic_agentic_memory.py`, `semantic_memory.py`, `external_memory.py` |
| [`Tools_and_Actions/`](Tools_and_Actions/README.md) | Tools that affect the world beyond text, building on the tool-calling foundation in `Core_Architecture/tool_use/`: code execution, web search, file editing, external API/MCP connectors, and browser automation | `code_interpreter.py`, `web_search.py`, `file_io_tools.py`, `api_connectors_mcp.py`, `browser_computer_use.py` |
| [`Multi_Agent_Systems/`](Multi_Agent_Systems/README.md) | Multiple agents working together: an orchestrator delegating to specialists, a reusable worker agent, permanent handoff, a validating supervisor, and a parallel uncoordinated swarm | `orchestrator.py`, `worker_agent.py`, `agent_handoff.py`, `supervisor_pattern.py`, `swarm.py` |
| [`Task_and_State_Management/`](Task_and_State_Management/README.md) | Managing the shape and progress of an agent's own work: task decomposition, a formally enforced state machine, context window management, and crash-resumable checkpointing | `task_decomposition.py`, `state_machine.py`, `pruning.py`, `summarization.py`, `retrieval.py`, `checkpointing.py` |
| [`Execution_Loops/`](Execution_Loops/README.md) | The core agentic loop and three ways to keep it under control: human approval gates, condition-based breakpoints, and an unconditional iteration cap | `basic_agentic_loop.py`, `human_in_the_loop.py`, `interrupts_breakpoints.py`, `max_iterations.py` |
| [`Safety_and_Control/`](Safety_and_Control/README.md) | Keeping an agent's actions bounded and reviewable: guardrails, sandboxed execution, role- and task-scoped permissions, and a durable audit trail | `guardrails.py`, `sandboxing.py`, `permission_scoping.py`, `audit_trail.py`, `minimal_footprint.py` |
| [`Agent_Frameworks_and_Patterns/`](Agent_Frameworks_and_Patterns/README.md) | Compositional patterns for structuring how an agent turns a request into a result: a tool registry, fixed prompt chaining, a classify-then-route agent, and a reusable output evaluator (reflection loop lives in `Planning_and_Reasoning/`) | `basic_tool_registry.py`, `basic_prompt_chaining.py`, `router_agent.py`, `evaluator_agent.py` |

Each directory's README explains the concept, what the code demonstrates, how to run it, example output, and the config knobs you can tune.
