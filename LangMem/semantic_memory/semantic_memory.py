"""
CONCEPT: Semantic memory — durable FACTS about a user, extracted from
conversation into a structured, schema-validated store, so they're
available in every future session, not just the one where they came up.

../../Memory/semantic_memory/semantic_memory.py builds this exact same idea by hand: a
tool Claude calls mid-conversation (`save_fact`), a hand-written JSON
Schema, and a plain JSON file on disk it reads/writes directly. This
template is the LangMem-native version — `create_memory_store_manager`
takes a Pydantic schema (`UserFact` below) and, given a batch of
conversation messages, decides on its own which facts are worth keeping,
extracts them as validated `UserFact` instances, and writes them into a
`BaseStore` — no hand-written save_fact tool, no manual JSON
read-modify-write. The MODEL still does the actual judgment call ("is
this worth remembering?"); what LangMem replaces is the plumbing around
that judgment (the tool schema, the dispatch, the file I/O), not the
judgment itself.

Namespacing (`("memories", "{customer_id}")`, resolved per call from
`config={"configurable": {"customer_id": ...}}`) gives every customer an
isolated fact store from ONE manager instance — the same per-session
isolation pattern as ../../LangChain/memory/memory.py's `session_id`, just for
facts that outlive a single conversation instead of a single session's
message history.

Use case: a support agent that extracts durable facts about each customer
after every message (email vs. phone preference, which department they
work in, product they use) into a per-customer store, reusing the
customer/support domain from ../../LangChain/prompt_templates/ and
../../LangChain/memory/. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys

import langmem
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"


class UserFact(BaseModel):
    """A durable fact about a customer — preference, role, or context
    worth remembering across every future conversation with them."""

    content: str = Field(description="The fact, written as a standalone sentence, e.g. 'Prefers email over phone calls.'")


INSTRUCTIONS = (
    "Extract durable facts worth remembering about this customer from the "
    "conversation: stated preferences, their role, their product, or "
    "context that would help a future conversation with them. Skip "
    "anything transient (the specific issue they're reporting right now)."
)


def build_manager(llm: BaseChatModel, store: BaseStore):
    # CONCEPT: namespace as a template — "{customer_id}" is filled in from
    # `config["configurable"]["customer_id"]` at invoke time, so one
    # manager instance serves every customer, each with an isolated slice
    # of the store, instead of building a new manager per customer.
    return langmem.create_memory_store_manager(
        llm,
        schemas=[UserFact],
        instructions=INSTRUCTIONS,
        namespace=("memories", "{customer_id}"),
        store=store,
    )


def remember(manager, customer_id: str, message_text: str) -> list[dict]:
    return manager.invoke(
        {"messages": [{"role": "user", "content": message_text}]},
        config={"configurable": {"customer_id": customer_id}},
    )


def recall(store: BaseStore, customer_id: str) -> list[str]:
    items = store.search(("memories", customer_id))
    return [item.value["content"]["content"] for item in items]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL)
    store = InMemoryStore()
    manager = build_manager(llm, store)

    print("Customer fact extraction (semantic memory demo). Type 'exit' to quit.")
    print("Each turn starts with a customer id — reuse one to keep accumulating facts about them.\n")
    print("Try: customer 'alex' -> \"I always prefer email, never call me\"\n")

    while True:
        customer_id = input("Customer id (or 'exit'): ").strip()
        if customer_id.lower() == "exit":
            print("Goodbye!")
            break
        if not customer_id:
            continue
        message = input(f"[{customer_id}] Message: ").strip()
        if not message:
            continue

        remember(manager, customer_id, message)
        facts = recall(store, customer_id)
        print(f"[{customer_id}] Facts on file: {facts}\n")


if __name__ == "__main__":
    main()
