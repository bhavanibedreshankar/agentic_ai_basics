"""
CONCEPT: Memory management tools — giving the AGENT ITSELF the ability to
decide when to save or search its own memories mid-conversation, as
ordinary tool calls, rather than the surrounding code deciding when
memory extraction happens.

../semantic_memory/ and ../episodic_memory/ both call a memory manager
AFTER every turn, unconditionally — the CODE decides "extract whatever's
worth remembering now," every time, whether or not anything memorable was
actually said. This template hands the model two tools instead —
`create_manage_memory_tool` (save/update/delete) and
`create_search_memory_tool` (semantic search) — bound on a
`create_agent` exactly the way ../../LangChain/agents_and_tools/agents_and_tools.py binds its
expense tools. The MODEL now decides, mid-conversation, whether this
message is worth saving and whether a question needs a memory search
first — the same "who decides" distinction
../../Core_Architecture/tool_use/basic_agentic_tools.py's docstring draws for tool use
in general, just applied to the agent's own memory instead of an external
action.

Use case: a support agent with `manage_memory`/`search_memory` tools over
a per-customer namespace, reusing the customer domain from
../semantic_memory/ and ../episodic_memory/. Ask it to remember something,
then in a LATER call (simulating a new session) ask it a question that
requires searching what it saved. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys

import langmem
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LangChain", "retrieval_augmented_generation"))
from retrieval_augmented_generation import HashEmbeddings  # noqa: E402 - see sys.path note above

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are a support agent. Use manage_memory to save anything worth "
    "remembering about this customer for future conversations (stated "
    "preferences, recurring issues). Use search_memory before answering "
    "questions that might depend on something you already know about them."
)


def build_store() -> BaseStore:
    return InMemoryStore(index={"dims": 64, "embed": HashEmbeddings()})


def build_agent(llm: BaseChatModel, store: BaseStore, customer_id: str):
    # CONCEPT: both tools are scoped to ONE customer's namespace at
    # construction time — the model can freely call manage_memory/
    # search_memory within that scope, but has no way to reach a
    # different customer's memories, the same hard isolation
    # ../semantic_memory/ and ../episodic_memory/ get from their
    # namespace-per-customer pattern, just enforced by tool scoping
    # instead of a config-supplied id.
    namespace = ("memories", customer_id)
    manage_tool = langmem.create_manage_memory_tool(namespace=namespace, store=store)
    search_tool = langmem.create_search_memory_tool(namespace=namespace, store=store)
    return create_agent(llm, [manage_tool, search_tool], system_prompt=SYSTEM_PROMPT)


def run_turn(agent, history: list) -> list:
    result = agent.invoke({"messages": history})
    new_messages = result["messages"]
    for message in new_messages[len(history):]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  [tool] {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"  [result] {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            print(f"\nClaude: {message.content}\n")
    return new_messages


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    store = build_store()

    print("Support agent with self-managed memory. Type 'exit' to quit.")
    print("Each turn starts with a customer id, so the agent's memory tools stay scoped to that customer.\n")
    print("Try: customer 'alex' -> \"Please always email me, never call\"")
    print("then a NEW session, same customer -> \"How should I contact you... I mean how should we contact you?\"\n")

    while True:
        customer_id = input("Customer id (or 'exit'): ").strip()
        if customer_id.lower() == "exit":
            print("Goodbye!")
            break
        message = input(f"[{customer_id}] You: ").strip()
        if not message:
            continue

        # CONCEPT: a fresh agent + fresh message history every turn here
        # deliberately simulates separate sessions — the memory tools are
        # the ONLY thing carrying anything over between them, since the
        # store (not the message history) is what's shared across the
        # `build_agent` calls below.
        agent = build_agent(llm, store, customer_id)
        run_turn(agent, [HumanMessage(content=message)])


if __name__ == "__main__":
    main()
