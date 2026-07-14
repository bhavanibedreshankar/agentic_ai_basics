"""
CONCEPT: Memory — letting a chain remember prior turns of a conversation,
keyed by a session id, instead of forgetting everything between calls.

A bare LCEL chain (see ../chains/chains.py) is STATELESS: `chain.invoke(x)`
only ever sees `x`, never anything from a previous call. That's fine for a
one-shot pipeline, but a chat needs the model to see what was already said.
`RunnableWithMessageHistory` wraps a chain so that, given a `session_id`,
it automatically fetches that session's prior messages, injects them into
the prompt via a `MessagesPlaceholder`, and appends the new turn back into
that same history after the call completes — all without the chain itself
holding any state.

This is the SAME core idea as ../../Memory/in_context_memory/in_context_memory.py — memory that
lives entirely in a Python process's RAM and vanishes when the process
exits, unlike ../../Memory/external_memory/ or ../../Memory/episodic_memory/, which persist to
disk on purpose. What's different here is the SCOPE: in_context_memory.py
has exactly one `messages` list for the whole program. This template keyed
by `session_id` supports many INDEPENDENT conversations at once — a
customer support chat handling several customers concurrently, say — each
with its own isolated history, without the calling code having to manage
separate `messages` lists by hand.

HONESTY NOTE: `RunnableWithMessageHistory` is marked deprecated as of
LangChain 1.x in favor of giving a LangGraph agent a checkpointer (see
../../LangGraph/persistence_and_checkpointing/), which persists the full
graph state, not just a message list, and survives process restarts. It's
kept here anyway because it isolates the MEMORY mechanic on its own,
without a graph or an agent loop around it — useful for understanding
what "add memory to a chain" means at its simplest, before layering on
everything LangGraph's checkpointer also does.

Use case: a support chat that can serve multiple customers in the same
running process. Switch `session_id` mid-demo to prove one customer's
history never leaks into another's. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise support agent. Remember details the customer already told you this session."),
        # CONCEPT: MessagesPlaceholder — a slot in the template that gets
        # filled with a LIST of prior messages, not a single string.
        # RunnableWithMessageHistory populates "history" automatically on
        # every call; you never build this list yourself.
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

# CONCEPT: the session store — a plain dict of session_id -> chat history
# object, live only for this process (see module docstring: this is
# in-memory, not persisted). A production app would swap this for a
# database-backed BaseChatMessageHistory implementation; RunnableWithMessageHistory
# doesn't care which one you use as long as get_session_history returns
# something implementing BaseChatMessageHistory.
_SESSION_STORE: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_chat_with_memory(llm: BaseChatModel) -> Runnable:
    """Wrap the chat chain with per-session history. Takes `llm` as a
    parameter so tests can substitute a fake model — see this directory's
    README for how that's verified without a real API key.
    """
    chain = CHAT_PROMPT | llm
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def send(chat_with_memory: Runnable, session_id: str, text: str) -> str:
    # CONCEPT: the session_id lives in `config`, not in the input dict —
    # it's routing metadata for RunnableWithMessageHistory, not part of
    # what actually gets sent to the model.
    response = chat_with_memory.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    chat_with_memory = build_chat_with_memory(llm)

    print("Multi-session support chat (memory demo). Type 'exit' to quit.")
    print("Each turn starts with a session id — reuse one to keep context, switch to isolate it.\n")
    print("Try: session 'alex' -> \"My order number is 4471\", then session 'jamie' -> \"What's my order number?\"\n")

    while True:
        session_id = input("Session id (or 'exit'): ").strip()
        if session_id.lower() == "exit":
            print("Goodbye!")
            break
        if not session_id:
            continue
        text = input(f"[{session_id}] You: ").strip()
        if not text:
            continue

        reply = send(chat_with_memory, session_id, text)
        print(f"[{session_id}] Claude: {reply}\n")


if __name__ == "__main__":
    main()
