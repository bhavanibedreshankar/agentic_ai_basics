"""
CONCEPT: Episodic memory — recalling a SPECIFIC PAST INTERACTION relevant
to what's happening right now, not just an extracted fact. "The last time
this customer had a billing issue, here's how it was resolved" needs the
whole episode (what happened, how it was fixed), retrieved by relevance
to the CURRENT situation — semantic search over stored experiences, not a
lookup by key.

../../Memory/episodic_memory/episodic_memory.py builds this by hand: a JSON file of past
episodes, and a plain bag-of-words scoring function the agent calls as a
tool to find the closest match. This template is the LangMem-native
version, and it's really ../semantic_memory/'s exact same
`create_memory_store_manager` mechanism aimed at a different schema:
where `UserFact` there captures a standalone fact, `Episode` here
captures `(situation, resolution)` — a whole past interaction — and
retrieval uses `store.search(..., query=...)`, LangGraph's SEMANTIC
search over the store's indexed field, rather than fact lookup by
namespace alone. The difference between ../semantic_memory/ and this file
is entirely in the SCHEMA and the retrieval mode, not the extraction
mechanism, which is worth noticing.

The store's semantic index reuses ../../LangChain/retrieval_augmented_generation/retrieval_augmented_generation.py's
`HashEmbeddings` — the same dependency-free word-hash-bucket embedding,
credited there — passed as `IndexConfig["embed"]` so `store.search()` can
rank past episodes by similarity to the current situation instead of
exact key lookup.

Use case: a support agent that recalls the closest-matching past episode
for a customer before responding to a new report, reusing the customer
domain from ../semantic_memory/. Type 'exit' to end the session.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LangChain", "retrieval_augmented_generation"))
from retrieval_augmented_generation import HashEmbeddings  # noqa: E402 - see sys.path note above

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"


class Episode(BaseModel):
    """A specific past interaction worth recalling when something similar
    comes up again."""

    situation: str = Field(description="What the customer reported or asked about")
    resolution: str = Field(description="How it was resolved")


INSTRUCTIONS = (
    "Extract the episode from this closed support ticket: what the "
    "customer originally reported (situation) and how it was ultimately "
    "resolved (resolution)."
)


def build_manager(llm: BaseChatModel, store: BaseStore):
    return langmem.create_memory_store_manager(
        llm,
        schemas=[Episode],
        instructions=INSTRUCTIONS,
        namespace=("episodes", "{customer_id}"),
        store=store,
    )


def build_store() -> BaseStore:
    # CONCEPT: `fields=["content.situation"]` tells the store to index
    # only the situation half of each stored Episode for similarity search
    # — a new ticket's TEXT should match past SITUATIONS, not past
    # resolutions. `create_memory_store_manager` wraps every stored value
    # as {"kind": <schema name>, "content": <the schema instance>}, hence
    # the "content." prefix.
    return InMemoryStore(index={"dims": 64, "embed": HashEmbeddings(), "fields": ["content.situation"]})


def close_ticket(manager, customer_id: str, ticket_transcript: str) -> None:
    manager.invoke(
        {"messages": [{"role": "user", "content": ticket_transcript}]},
        config={"configurable": {"customer_id": customer_id}},
    )


def recall_similar_episode(store: BaseStore, customer_id: str, new_situation: str) -> dict | None:
    results = store.search(("episodes", customer_id), query=new_situation, limit=1)
    if not results or results[0].score is None or results[0].score <= 0:
        return None
    return results[0].value["content"]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL)
    store = build_store()
    manager = build_manager(llm, store)

    print("Episodic recall for repeat customers (episodic memory demo). Type 'exit' to quit.")
    print("First close a ticket to build up episode history, then submit a similar new one.\n")
    print("Try: customer 'alex', close a ticket about \"duplicate charge on invoice, refunded it\",")
    print("then a new ticket: \"I was charged twice again this month\"\n")

    while True:
        customer_id = input("Customer id (or 'exit'): ").strip()
        if customer_id.lower() == "exit":
            print("Goodbye!")
            break
        if not customer_id:
            continue
        action = input("(c)lose a past ticket or (n)ew ticket? ").strip().lower()
        text = input("Text: ").strip()
        if not text:
            continue

        if action.startswith("c"):
            close_ticket(manager, customer_id, text)
            print("  [episode recorded]\n")
        else:
            episode = recall_similar_episode(store, customer_id, text)
            if episode:
                print(f"  [similar past episode found] situation: {episode['situation']!r}")
                print(f"  [it was resolved by] {episode['resolution']}\n")
            else:
                print("  [no similar past episode found for this customer]\n")


if __name__ == "__main__":
    main()
