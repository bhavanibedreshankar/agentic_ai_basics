"""
CONCEPT: Background memory consolidation — the hot-path vs. background
distinction. ../semantic_memory/ and ../memory_management_tools/ both
extract/save memory SYNCHRONOUSLY, in the middle of responding: the user
waits for the extraction call to finish before they get a reply. LangMem's
`ReflectionExecutor` runs the same kind of extraction ASYNCHRONOUSLY,
after the fact, in a background thread — the user never waits for it —
and, more importantly, DEBOUNCES rapid-fire triggers: submitting a new
reflection for the same `thread_id` before the pending one has run
CANCELS the pending one and replaces it, so a burst of messages in the
same conversation triggers exactly ONE consolidation pass instead of one
per message.

This is genuinely runnable end to end in this environment —
`ReflectionExecutor(reflector, store=store)` with a plain `Runnable`
reflector (no server, no URL) returns a `LocalReflectionExecutor` backed
by a real Python `threading.Thread` and `queue.PriorityQueue`, verified
directly for this file: submitting twice for the same `thread_id` within
the delay window left the FIRST submission's `Future.cancelled() ==
True` and only the second one's extraction actually ran (confirmed by
counting model calls). No hypothetical infrastructure needed — this
really is what a deployed setup does, just single-process instead of
scaled out.

Use case: a burst of three quick messages in one support conversation,
submitted for background consolidation as they arrive — only the LAST
submission's reflection actually runs, shown side by side with what the
../semantic_memory/-style hot-path equivalent (extracting after every
single message, blocking) would have cost instead. Run directly; not
interactive (the timing/debounce behavior is easiest to see against a
fixed, known sequence rather than live typed input).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future

import langmem
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langmem import ReflectionExecutor
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LangChain", "retrieval_augmented_generation"))
from retrieval_augmented_generation import HashEmbeddings  # noqa: E402 - see sys.path note above

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"

CONSOLIDATION_DELAY_SECONDS = 2  # how long a submission waits before it actually runs


class UserFact(BaseModel):
    """Same schema as ../semantic_memory/'s UserFact — this template
    changes WHEN extraction runs, not what gets extracted."""

    content: str = Field(description="A durable fact about the customer")


def build_store() -> BaseStore:
    return InMemoryStore(index={"dims": 64, "embed": HashEmbeddings()})


def build_manager(llm: BaseChatModel, store: BaseStore, customer_id: str):
    return langmem.create_memory_store_manager(
        llm,
        schemas=[UserFact],
        instructions="Extract durable facts worth remembering about this customer.",
        namespace=("memories", customer_id),
        store=store,
    )


def build_executor(manager, store: BaseStore) -> ReflectionExecutor:
    # CONCEPT: this one call turns a synchronous extraction Runnable into
    # something you SUBMIT rather than call directly — the manager's own
    # invoke() logic is completely unchanged; only how/when it runs
    # changes.
    return ReflectionExecutor(manager, store=store)


def submit_message(executor: ReflectionExecutor, thread_id: str, message_text: str) -> Future:
    config = {"configurable": {"thread_id": thread_id}}
    # CONCEPT: submitting again with the SAME thread_id before an earlier
    # submission has fired cancels it — see LocalReflectionExecutor.submit
    # in the installed langmem source for the exact cancel-and-replace
    # logic this relies on.
    return executor.submit(
        {"messages": [{"role": "user", "content": message_text}]},
        config=config,
        after_seconds=CONSOLIDATION_DELAY_SECONDS,
        thread_id=thread_id,
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL)
    store = build_store()
    manager = build_manager(llm, store, customer_id="alex")
    executor = build_executor(manager, store)

    print("Background memory consolidation demo (no interactive input — a fixed message burst).\n")

    burst = [
        "I prefer email.",
        "Actually, email or text is fine.",
        "On second thought, just always email me, never call.",
    ]

    futures = []
    for i, message in enumerate(burst, start=1):
        print(f"[{i}/{len(burst)}] message arrives: {message!r} — submitting for background consolidation")
        future = submit_message(executor, thread_id="conv-alex-1", message_text=message)
        futures.append(future)
        time.sleep(0.3)  # arrives faster than CONSOLIDATION_DELAY_SECONDS, so each resubmission cancels the last

    print(f"\nOf {len(futures)} submissions, {sum(f.cancelled() for f in futures)} were cancelled by a later one.")
    print("Waiting for the final (uncancelled) submission to actually run...")

    final_result = futures[-1].result(timeout=CONSOLIDATION_DELAY_SECONDS + 5)
    print(f"\nConsolidation ran once, on the FINAL message only: {final_result}")

    print("\nCompare: ../semantic_memory/'s hot-path equivalent would have called the model once PER message above")
    print("(3 calls), blocking the reply each time, instead of once in the background after the burst settled.")

    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
