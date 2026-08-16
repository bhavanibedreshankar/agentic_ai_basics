# Agentic AI Basics

A learning series of small, self-contained Python templates for building with the Claude API — each one focused on a single agentic AI concept, with comments explaining the "why" as much as the "what". Every template lives in its own directory with a dedicated README going into that concept in depth; this file just covers setup and a quick index of everything here.

**[→ Browse the visual index](https://bhavanibedreshankar.github.io/agentic_ai_basics/)** — a field-reference landing page covering all 51 templates.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

Don't have a key? Create one at [platform.claude.com](https://platform.claude.com).

Each template is run from the repo root, e.g.:

```bash
python3 01_Core_Architecture/basics/basic.py
```

Most templates are interactive chat loops — type `exit` to end the conversation.

## Templates

Numbered as a suggested learning path — start at 1 and work down, or jump straight to whatever you need. Each row builds on concepts from the rows above it.

| # | Directory | Concept | File(s) |
|---|---|---|---|
| 1 | [`01_Core_Architecture/`](01_Core_Architecture/README.md) | What an agent is actually made of: a single request/response call, token usage tracking, the reasoning-engine LLM, tool use/function calling, the system prompt that shapes it all, and that same agent rebuilt as a modular, GAME-style (Goals/Actions/Memory/Environment) package | `basic.py`, `basic_token_tracking.py`, `agent.py`, `llm_backbone.py`, `basic_agentic_tools.py`, `system_prompt.py`, `modular_agent/` (`goals.py`, `memory.py`, `actions.py`, `environment.py`, `language.py`, `loop.py`, `agent.py`, `builtin_actions.py`, `main.py`) |
| 2 | [`02_Execution_Loops/`](02_Execution_Loops/README.md) | The core agentic loop and three ways to keep it under control: human approval gates, condition-based breakpoints, and an unconditional iteration cap | `basic_agentic_loop.py`, `human_in_the_loop.py`, `interrupts_breakpoints.py`, `max_iterations.py` |
| 3 | [`03_Tools_and_Actions/`](03_Tools_and_Actions/README.md) | Tools that affect the world beyond text, building on the tool-calling foundation in `01_Core_Architecture/tool_use/`: code execution, web search, file editing, external API/MCP connectors, and browser automation | `code_interpreter.py`, `web_search.py`, `file_io_tools.py`, `api_connectors_mcp.py`, `browser_computer_use.py` |
| 4 | [`04_Planning_and_Reasoning/`](04_Planning_and_Reasoning/README.md) | Getting Claude to reason more reliably: ReAct, chain-of-thought, tree of thought, plan-and-execute, and self-reflection | `react_agent.py`, `chain_of_thought.py`, `tree_of_thought.py`, `plan_and_execute.py`, `self_reflection.py` |
| 5 | [`05_Memory/`](05_Memory/README.md) | Six kinds of agent memory: in-context, working, episodic, basic long-term facts, structured semantic, and external/vector store | `in_context_memory.py`, `working_memory.py`, `episodic_memory.py`, `basic_agentic_memory.py`, `semantic_memory.py`, `external_memory.py` |
| 6 | [`06_Task_and_State_Management/`](06_Task_and_State_Management/README.md) | Managing the shape and progress of an agent's own work: task decomposition, a formally enforced state machine, context window management, and crash-resumable checkpointing | `task_decomposition.py`, `state_machine.py`, `pruning.py`, `summarization.py`, `retrieval.py`, `checkpointing.py` |
| 7 | [`07_Multi_Agent_Systems/`](07_Multi_Agent_Systems/README.md) | Multiple agents working together: an orchestrator delegating to specialists, a reusable worker agent, permanent handoff, a validating supervisor, and a parallel uncoordinated swarm | `orchestrator.py`, `worker_agent.py`, `agent_handoff.py`, `supervisor_pattern.py`, `swarm.py` |
| 8 | [`08_Agent_Frameworks_and_Patterns/`](08_Agent_Frameworks_and_Patterns/README.md) | Compositional patterns for structuring how an agent turns a request into a result: a tool registry, native tool search over a large tool catalog, fixed prompt chaining, runtime prompt assembly from conditional sections, a classify-then-route agent, and a reusable output evaluator (reflection loop lives in `04_Planning_and_Reasoning/`) | `basic_tool_registry.py`, `dynamic_tool_selection.py`, `basic_prompt_chaining.py`, `dynamic_prompt_construction.py`, `router_agent.py`, `evaluator_agent.py` |
| 9 | [`09_RAG_and_Knowledge/`](09_RAG_and_Knowledge/README.md) | Retrieval-Augmented Generation and its building blocks: embedding, chunking, hybrid dense+sparse search, coarse-to-fine retrieval, graph RAG (fixed-pipeline and agentic), and corrective/self-reflective RAG | `embedding_search.py`, `chunking_strategies.py`, `basic_rag.py`, `hybrid_search.py`, `coarse_to_fine_retrieval.py`, `graph_construction.py`, `graph_retrieval.py`, `graph_rag_agent.py`, `corrective_rag.py` |
| 10 | [`10_Safety_and_Control/`](10_Safety_and_Control/README.md) | Keeping an agent's actions bounded and reviewable: guardrails, sandboxed execution, role- and task-scoped permissions, and a durable audit trail | `guardrails.py`, `sandboxing.py`, `permission_scoping.py`, `audit_trail.py`, `minimal_footprint.py` |
| 11 | [`11_Caching/`](11_Caching/README.md) | Keeping only what's worth paying to keep: server-side prompt caching of a stable prefix, scoring incoming items for importance before they ever enter context, bounding a client-side cache with LRU/TTL eviction, and memoizing an expensive tool call's result | `context_caching.py`, `selective_context_retention.py`, `cache_eviction_policies.py`, `tool_result_caching.py` |
| 12 | [`12_Model_Routing/`](12_Model_Routing/README.md) | Picking which model tier actually answers a request so simple tasks never pay for an expensive model: classify-then-route, free heuristics plus a session budget, adaptive escalation on low confidence, and failover to another model on a real API error | `task_classifier_router.py`, `cost_aware_model_selection.py`, `complexity_based_escalation.py`, `multi_model_fallback.py` |
| 13 | [`13_Benchmarking/`](13_Benchmarking/README.md) | Measuring an agent systematically: deterministic accuracy scoring, LLM-as-judge scoring for open-ended output, latency/cost profiling across effort levels, regression testing against a stored baseline, side-by-side model/prompt comparison, and scoring a whole tool-use trace rather than just the final answer | `task_accuracy_eval.py`, `llm_judge_benchmarking.py`, `latency_cost_benchmarking.py`, `regression_testing.py`, `model_prompt_comparison.py`, `trace_evaluation.py` |
| 14 | [`14_Agent_Testing/`](14_Agent_Testing/README.md) | The software-testing discipline for an agent's code, distinct from `13_Benchmarking/`'s quality scoring: unit-testing tools in isolation, integration-testing the orchestration loop against a stubbed LLM, adversarial/red-team testing against a growing attack corpus, shared fixture/mock infrastructure, and layering it all into one cost-aware CI pyramid as the agent scales | `tool_unit_testing.py`, `agent_integration_testing.py`, `adversarial_safety_testing.py`, `fixture_and_mock_management.py`, `test_suite_pyramid.py` |
| 15 | [`15_Self_Evolving_Agents/`](15_Self_Evolving_Agents/README.md) | An agent that closes the loop on its own instructions: negative feedback on an answer gets distilled into a rule, spliced into its system prompt, and persisted so the very next call — even in a new process — already reflects it | `self_evolving_agents.py` |
| 16 | [`16_Dynamic_Agent_Spawning/`](16_Dynamic_Agent_Spawning/README.md) | A meta-agent with no built-in specialists: it invents a sub-agent's role, persona, and system prompt at runtime and assigns it a task, spawning as many uniquely-defined specialists as a request needs (capped per turn) | `dynamic_agent_spawning.py` |
| 17 | [`17_LangChain/`](17_LangChain/README.md) | The LangChain framework layer built on top of the raw Claude API: reusable prompt templates, LCEL chains, per-session memory, a prebuilt tool-calling agent, retrieval-augmented generation, structured output parsing, a lightweight LangGraph intro, and cross-cutting callback tracing | `prompt_templates.py`, `chains.py`, `memory.py`, `agents_and_tools.py`, `retrieval_augmented_generation.py`, `output_parsers.py`, `langgraph_workflows.py`, `callbacks_and_tracing.py` |
| 18 | [`18_LangGraph/`](18_LangGraph/README.md) | The graph layer under LangChain's `create_agent` and deprecated `RunnableWithMessageHistory`: state-merging reducers, cycles, checkpointed persistence, human-in-the-loop pause/resume, streaming, and multi-agent subgraph composition | `state_graph_basics.py`, `conditional_routing.py`, `persistence_and_checkpointing.py`, `human_in_the_loop.py`, `streaming.py`, `multi_agent_subgraphs.py` |
| 19 | [`19_LangMem/`](19_LangMem/README.md) | Long-term agent memory, distinct from LangChain's per-session history: extracting durable facts and whole past episodes into a structured store, an agent's own prompt improving from feedback, letting the model manage its own memory via tools, and deferring extraction to a debounced background pass | `semantic_memory.py`, `episodic_memory.py`, `procedural_memory.py`, `memory_management_tools.py`, `background_memory_consolidation.py` |

Each directory's README explains the concept, what the code demonstrates, how to run it, example output, and the config knobs you can tune.

### Why this order

1-3 are the load-bearing fundamentals: what an agent is, the loop that drives it, and giving it real tools. 4-8 build the agent's internal sophistication — better reasoning, memory across turns, managing its own long-running work, coordinating multiple agents, and reusable structural patterns. 9-12 are cross-cutting infrastructure concerns (knowledge retrieval, safety, cost) that apply once an agent is doing real work. 13-14 are how you know any of the above still works as you keep changing it. 15-16 are more experimental/advanced patterns (agents that rewrite their own instructions or spawn specialists on the fly) best understood after the fundamentals are solid. 17-19 are the 17_LangChain/18_LangGraph/LangMem ecosystem — the same core concepts from 1-12, revisited through a specific framework's abstractions, so they're deliberately last.
