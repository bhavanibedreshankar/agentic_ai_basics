"""
CONCEPT: Prompt Templates — LangChain's abstraction for a reusable,
parameterized prompt. The instructions are written ONCE; only the variable
slots (customer name, product, ticket text) change on each call. LangChain
validates at construction time that every `{placeholder}` in the template
text has a matching input variable, and raises immediately if you forget
one when invoking — a whole class of "silently sent a prompt with a
literal '{ticket_text}' in it" bugs simply can't happen.

Every other template in this repo builds its prompt by hand — an f-string
system prompt plus a plain string user message, e.g.
../../Core_Architecture/basics/basic.py's `SYSTEM_PROMPT` constant and
../../Safety_and_Control/audit_trail/audit_trail.py's inline
`f"..."` construction. That works, but the template and the data are
tangled together in one string, and there's no validation that every slot
got filled before the request goes out. `ChatPromptTemplate` separates the
two: a template object you define once, and a dict of variables you supply
per call — the same separation ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py
achieves by hand with separate SYSTEM_PROMPT constants per step, but here
it's a first-class object with its own validation and composition methods
(see PARTIAL_TRIAGE_PROMPT below, and `chain.py` in this same topic for
composing a template directly into a runnable pipeline).

Use case: a support-ticket triage assistant. The SAME template is reused
across every ticket — only customer_name, product, and ticket_text change
per call — and a second, partially-filled template shows how a constant
(the support tier policy) can be baked into the template once instead of
threaded through every call site. Type 'exit' to end the session.
"""

from __future__ import annotations

import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# CONCEPT: ChatPromptTemplate — a sequence of (role, template-string) pairs.
# Each template string can contain {placeholders}; LangChain parses them out
# of the text automatically and exposes them as `.input_variables`, so you
# can inspect what a template expects before ever calling it.
# ---------------------------------------------------------------------------
TRIAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support ticket triage assistant. Read the ticket and "
            "respond with: a one-line summary, a priority (low/medium/high/"
            "urgent), and the team it should route to (billing, technical, "
            "or account). Be concise — three lines total, no preamble.",
        ),
        (
            "human",
            "Customer: {customer_name}\nProduct: {product}\nTicket:\n{ticket_text}",
        ),
    ]
)
print(f"[TRIAGE_PROMPT expects: {TRIAGE_PROMPT.input_variables}]")

# ---------------------------------------------------------------------------
# CONCEPT: partial variables — filling in SOME of a template's slots at
# definition time, leaving the rest for call time. `support_tier` is a
# constant for this whole session (read once, e.g. from a config file); it
# would be wasteful and error-prone to make every call site pass it in
# alongside the genuinely per-ticket fields. Note PARTIAL_TRIAGE_PROMPT's
# `.input_variables` below no longer includes "support_tier" — it's already
# baked in.
# ---------------------------------------------------------------------------
SUPPORT_TIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support ticket triage assistant for a {support_tier} "
            "tier customer base. {support_tier} customers get a stricter "
            "priority bar — only true outages count as 'urgent'. Respond "
            "with a one-line summary, a priority, and a routing team.",
        ),
        ("human", "Customer: {customer_name}\nProduct: {product}\nTicket:\n{ticket_text}"),
    ]
)
PARTIAL_TRIAGE_PROMPT = SUPPORT_TIER_PROMPT.partial(support_tier="enterprise")
print(f"[PARTIAL_TRIAGE_PROMPT still expects: {PARTIAL_TRIAGE_PROMPT.input_variables}]")

# ---------------------------------------------------------------------------
# CONCEPT: PromptTemplate — the plain-text counterpart to ChatPromptTemplate,
# for backends or steps that just want one string rather than a role/content
# message list. Useful for the non-chat pieces of a pipeline, e.g. a
# summarization step that doesn't need conversational roles.
# ---------------------------------------------------------------------------
ONE_LINE_SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize this support ticket in exactly one sentence:\n\n{ticket_text}"
)


def build_triage_chain(llm: BaseChatModel) -> Runnable:
    """Compose the template with a chat model into a runnable pipeline.

    Takes `llm` as a parameter (rather than closing over a module-level
    global) specifically so tests can substitute a fake model — see
    prompt_templates_test in this file's README for how that's verified
    without a real API key.
    """
    return TRIAGE_PROMPT | llm


def triage_ticket(chain: Runnable, customer_name: str, product: str, ticket_text: str) -> str:
    # CONCEPT: calling a template-backed chain is just `.invoke(dict)` — the
    # dict's keys must match `TRIAGE_PROMPT.input_variables` exactly, or
    # LangChain raises a KeyError before ever contacting the model.
    response = chain.invoke(
        {"customer_name": customer_name, "product": product, "ticket_text": ticket_text}
    )
    return response.content


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    chain = build_triage_chain(llm)

    print("Support ticket triage (prompt templates demo). Type 'exit' to quit.\n")
    print("Try: product=\"mobile app\", ticket=\"App crashes every time I open the camera tab\"\n")

    while True:
        product = input("Product (or 'exit'): ").strip()
        if product.lower() == "exit":
            print("Goodbye!")
            break
        ticket_text = input("Ticket text: ").strip()
        if not ticket_text:
            continue

        result = triage_ticket(chain, customer_name="you", product=product, ticket_text=ticket_text)
        print(f"\nTriage:\n{result}\n")


if __name__ == "__main__":
    main()
