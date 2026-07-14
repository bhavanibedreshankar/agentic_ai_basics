"""
CONCEPT: Streaming — watching a graph's progress node by node as it runs,
instead of only getting the final result once everything finishes.

Every other template in this repo, LangGraph ones included, calls
`.invoke()` and waits for the whole thing to complete before printing
anything about what happened internally (see e.g.
../state_graph_basics/state_graph_basics.py, which only prints its audit
log AFTER `graph.invoke()` returns). `.stream()` is the same graph, same
nodes, same edges — just observed differently: it's a generator that
yields a value after each node finishes, so a caller (e.g. a UI showing
"searching... summarizing... done") can react to progress in real time
instead of staring at a blank screen until the whole run is over.

This template shows the two most immediately useful `stream_mode` values:
  - `"updates"` — yields ONLY what each node just changed, as
    `{node_name: {changed_fields}}`. Good for a progress indicator: "step
    'summarize' just finished, here's what it produced."
  - `"values"` — yields the FULL accumulated state after each node,
    including the very first snapshot before any node has run. Good when
    a caller wants the whole current picture at every step, not just the
    diff.

Both stream the exact same 3-node pipeline — nothing about the graph
itself changes between the two; only which shape of update you ask for.

Use case: a document analysis pipeline (summarize -> extract keywords ->
classify sentiment) over a paragraph of text, run once with each stream
mode so the difference in what gets yielded is directly visible. Type
'exit' to end the session.
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512


class DocumentState(TypedDict):
    text: str
    summary: str
    keywords: str
    sentiment: str


def summarize(state: DocumentState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"Summarize in one sentence:\n\n{state['text']}")
    return {"summary": reply.content}


def extract_keywords(state: DocumentState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"List 3-5 keywords from this text, comma-separated:\n\n{state['text']}")
    return {"keywords": reply.content}


def classify_sentiment(state: DocumentState, llm: BaseChatModel) -> dict:
    reply = llm.invoke(f"Classify the sentiment as one word (positive/neutral/negative):\n\n{state['text']}")
    return {"sentiment": reply.content}


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(DocumentState)
    graph.add_node("summarize", lambda state: summarize(state, llm))
    graph.add_node("extract_keywords", lambda state: extract_keywords(state, llm))
    graph.add_node("classify_sentiment", lambda state: classify_sentiment(state, llm))

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "extract_keywords")
    graph.add_edge("extract_keywords", "classify_sentiment")
    graph.add_edge("classify_sentiment", END)

    return graph.compile()


def stream_updates(graph: CompiledStateGraph, text: str) -> list[dict]:
    """Collect every chunk `.stream(..., stream_mode="updates")` yields —
    one {node_name: {changed_fields}} dict per finished node.
    """
    chunks = []
    for chunk in graph.stream({"text": text, "summary": "", "keywords": "", "sentiment": ""}, stream_mode="updates"):
        chunks.append(chunk)
        node_name, update = next(iter(chunk.items()))
        print(f"  [updates] {node_name} -> {update}")
    return chunks


def stream_values(graph: CompiledStateGraph, text: str) -> list[dict]:
    """Collect every chunk `.stream(..., stream_mode="values")` yields —
    the full state snapshot after each node, starting with the initial
    state before any node has run.
    """
    chunks = []
    for chunk in graph.stream({"text": text, "summary": "", "keywords": "", "sentiment": ""}, stream_mode="values"):
        chunks.append(chunk)
        print(f"  [values] {chunk}")
    return chunks


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    graph = build_graph(llm)

    print("Document analysis graph (streaming demo). Type 'exit' to quit.\n")
    print("Try: \"Our new feature launch exceeded every target we set for the quarter.\"\n")

    while True:
        text = input("Text (or 'exit'): ").strip()
        if text.lower() == "exit":
            print("Goodbye!")
            break
        if not text:
            continue

        print("\n-- stream_mode='updates' --")
        stream_updates(graph, text)

        print("\n-- stream_mode='values' --")
        stream_values(graph, text)
        print()


if __name__ == "__main__":
    main()
