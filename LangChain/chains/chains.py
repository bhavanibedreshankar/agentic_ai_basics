"""
CONCEPT: Chains — composing prompts, models, and plain Python logic into a
single pipeline using LangChain Expression Language (LCEL). The `|`
operator wires steps together the way a Unix pipe wires processes: each
step's output becomes the next step's input, and the whole pipeline is
itself a `Runnable` you can `.invoke()`, `.stream()`, or `.batch()` as one
unit.

../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py builds the same KIND of
thing — a fixed sequence of LLM calls with a programmatic gate — but by
hand: separate Python functions, a manually-checked `if not passed: return
None`, and a plain string threaded between calls. This template rebuilds
that idea with LCEL's own vocabulary, which buys three things prompt_chaining.py's
raw-function version doesn't get for free:
  - `RunnableParallel` runs independent branches CONCURRENTLY, not just
    sequentially — the summary and sentiment steps below both depend only
    on the original review text, so there's no reason to wait for one
    before starting the other.
  - `RunnableBranch` is a first-class conditional step IN the pipeline
    (condition, runnable) pairs plus a default — the gate isn't an `if`
    statement bolted onto the outside of the chain, it's part of the chain.
  - `StrOutputParser` standardizes "just give me the text" instead of every
    step manually doing `"".join(b.text for b in response.content if ...)`
    the way every raw-`anthropic`-SDK template in this repo does.

Use case: a product review pipeline. Summary and sentiment are computed in
parallel from the same review text; a branch then either drafts a reply
(non-negative sentiment) or escalates to a human agent (negative) instead
of auto-replying. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableParallel, RunnablePassthrough

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [("human", "Summarize this product review in one short sentence:\n\n{review_text}")]
)
SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Classify the sentiment of this product review as exactly one "
            "word — positive, neutral, or negative — with no other text:\n\n{review_text}",
        )
    ]
)
REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write brief, warm replies to product reviews on behalf of "
            "the company. Two to three sentences, no corporate jargon.",
        ),
        ("human", "Review: {review_text}\nSummary: {summary}\nSentiment: {sentiment}"),
    ]
)


def escalate(fields: dict) -> str:
    # CONCEPT: the "reply" branch and this "escalate" branch have the SAME
    # input shape ({review_text, summary, sentiment}) and are interchangeable
    # from RunnableBranch's point of view — it just picks one Runnable over
    # another based on the condition, same as an `if/else` picking a
    # function to call, just expressed as pipeline data instead of control
    # flow around the pipeline.
    return f"[escalated to a human agent] summary: {fields['summary']} (sentiment: {fields['sentiment']})"


def build_review_chain(llm: BaseChatModel) -> Runnable:
    """Compose the full pipeline. Takes `llm` as a parameter so tests can
    substitute a fake model — see this directory's README for how that's
    verified without a real API key.
    """
    # CONCEPT: RunnableParallel — summary and sentiment each only need
    # {review_text}, so they run as independent branches over the SAME
    # input rather than one waiting on the other. RunnablePassthrough()
    # carries the original input dict through unchanged, so review_text is
    # still available to the branch step below even though neither the
    # summary nor sentiment prompt outputs it.
    analyze = RunnableParallel(
        summary=SUMMARY_PROMPT | llm | StrOutputParser(),
        sentiment=SENTIMENT_PROMPT | llm | StrOutputParser(),
        review_text=RunnablePassthrough(),
    )

    # CONCEPT: RunnableBranch — a sequence of (condition, runnable) pairs,
    # evaluated in order, with a final default runnable if none match. This
    # is the LCEL-native equivalent of prompt_chaining.py's
    # `if not passed: return None` gate, but living INSIDE the pipeline
    # rather than as an early return wrapped around it.
    respond = RunnableBranch(
        (lambda fields: fields["sentiment"].strip().lower() == "negative", escalate),
        REPLY_PROMPT | llm | StrOutputParser(),  # default: draft a reply
    )

    # CONCEPT: the `|` operator (RunnableSequence) chains analyze -> respond
    # into a single Runnable — this is the whole pipeline, invoked as one
    # unit with `.invoke({"review_text": ...})`.
    return analyze | respond


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    chain = build_review_chain(llm)

    print("Product review responder (LCEL chains demo). Type 'exit' to quit.\n")
    print("Try: \"The battery life is terrible and it stopped charging after a week.\"\n")

    while True:
        review_text = input("Review: ").strip()
        if review_text.lower() == "exit":
            print("Goodbye!")
            break
        if not review_text:
            continue

        result = chain.invoke({"review_text": review_text})
        print(f"\n{result}\n")


if __name__ == "__main__":
    main()
