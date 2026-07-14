# LangGraph

The graph-orchestration library `LangChain/agents_and_tools/agents_and_tools.py`'s `create_agent` is built on, explored directly. [`../LangChain/langgraph_workflows/`](../LangChain/langgraph_workflows/README.md) is the entry point into this topic — a single 3-node branching graph shown from inside the LangChain topic; everything here goes deeper into the primitives that file uses without dwelling on: state-merge reducers, cycles, checkpointed persistence, first-class pause/resume, incremental observation, and subgraph composition.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`state_graph_basics/`](state_graph_basics/README.md) | How state updates from multiple nodes MERGE (`Annotated[..., operator.add]` reducers) |
| 2 | [`conditional_routing/`](conditional_routing/README.md) | Edges that can CYCLE, not just branch — a graph-native retry loop with a safety cap |
| 3 | [`persistence_and_checkpointing/`](persistence_and_checkpointing/README.md) | A graph's full state surviving across separate `.invoke()` calls, keyed by `thread_id` |
| 4 | [`human_in_the_loop/`](human_in_the_loop/README.md) | Pausing execution for a human decision (`interrupt()`), resumable because of #3's checkpointer |
| 5 | [`streaming/`](streaming/README.md) | Observing a graph run incrementally (`.stream()`) instead of only getting the final state |
| 6 | [`multi_agent_subgraphs/`](multi_agent_subgraphs/README.md) | Composing a compiled graph as a NODE inside a larger graph |

## Setup

Same as the rest of the repo, plus LangGraph:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
pip install langchain langchain-core langchain-anthropic langgraph pydantic
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 LangGraph/state_graph_basics/state_graph_basics.py
```

## How these relate to each other

| | Adds over a plain StateGraph | Needs a checkpointer? |
|---|---|---|
| `state_graph_basics/` | Reducer-based state merging | No |
| `conditional_routing/` | Cycles — an edge that routes back to an earlier node | No |
| `persistence_and_checkpointing/` | State surviving across separate `.invoke()` calls | Yes — this IS the checkpointer |
| `human_in_the_loop/` | Pausing mid-node for a human decision | Yes — the paused state has to be saved somewhere |
| `streaming/` | Incremental visibility into execution, not new graph capability | No |
| `multi_agent_subgraphs/` | Composing compiled graphs as nodes | No |

`state_graph_basics/` and `conditional_routing/` are the two structural primitives everything else builds on: how state merges, and how edges can loop. `persistence_and_checkpointing/` adds durability across calls, which `human_in_the_loop/` then depends on directly — an `interrupt()` with no checkpointer has nowhere to save the paused state. `streaming/` is orthogonal to all of the above — it changes how you OBSERVE a run, not what the graph can do. `multi_agent_subgraphs/` is the odd one out structurally: instead of adding a new node/edge primitive, it shows that a NODE can itself be an entire other compiled graph — and, per that directory's HONESTY NOTE, exposes a real limitation of `state_graph_basics/`'s reducer pattern once a subgraph boundary is involved.
