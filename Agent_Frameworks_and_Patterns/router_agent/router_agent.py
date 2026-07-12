"""
CONCEPT: Router Agent — an agent that classifies the input and routes it
to the appropriate sub-agent or tool, as a single UPFRONT decision made
BEFORE any response generation begins.

This is a mechanically different routing pattern from
`../../Multi_Agent_Systems/agent_handoff/agent_handoff.py`, even though
both end up sending a request to a different specialist:
  - agent_handoff routes via TOOL USE, mid-conversation — the model
    itself decides, while already generating a response, to call a
    `transfer_to_X` tool. Routing there is a choice the MODEL makes,
    inside its own agentic loop, and it can happen after some back-and-
    forth has already occurred.
  - A router agent classifies FIRST, via a single structured-output
    call, and only THEN dispatches to a handler — no tool-calling loop
    is involved in the routing decision itself. The classification and
    the eventual response are two separate, sequential API calls (the
    same "call 1 feeds call 2" shape as
    `../prompt_chaining/basic_prompt_chaining.py`), not one model
    choosing mid-turn to redirect itself.

Because classification is a narrow, single-purpose call, it's also a
good candidate for a cheaper/faster model than the one that generates
the final response — this template keeps everything on one model for
consistency with the rest of the repo, but the two calls being
independent is exactly what would let you swap the classifier to a
smaller model in a real system without touching the handlers at all.

The classifier returns a category from a FIXED enum via structured
output — never free text to parse — plus a confidence flag; a low-
confidence classification falls back to a general handler instead of
committing to a possibly-wrong specialist.

Use case: a support-ticket router classifying incoming messages into
billing / technical / sales / general, each handled by its own focused
call. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

CATEGORIES = ["billing", "technical", "sales", "general"]

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the support message into exactly one category: billing, "
    "technical, sales, or general. Also report your confidence — set "
    "confident to false if the message could plausibly belong to more "
    "than one category or doesn't clearly fit any of them."
)

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "confident": {"type": "boolean"},
    },
    "required": ["category", "confident"],
    "additionalProperties": False,
}

HANDLER_SYSTEM_PROMPTS = {
    "billing": "You are a billing support specialist. Help with invoices, payments, and refunds.",
    "technical": "You are a technical support specialist. Help debug errors and troubleshoot issues.",
    "sales": "You are a sales specialist. Answer questions about plans, pricing, and upgrades.",
    "general": "You are a general support agent. Handle anything that doesn't clearly fit billing, technical, or sales.",
}


def classify(message: str) -> tuple[str, bool]:
    """CONCEPT: the router's entire job — one structured-output call
    returning a category from a closed set, never free text to parse or
    guess at. Returns (category, confident).
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=CLASSIFY_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": CLASSIFY_SCHEMA}},
        messages=[{"role": "user", "content": message}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    return result["category"], result["confident"]


def route(message: str) -> tuple[str, str]:
    """CONCEPT: classify, then dispatch — two separate calls, not one
    tool-calling loop. Returns (category_used, response_text). Falls back
    to 'general' when the classifier itself is unsure, rather than
    forwarding a low-confidence guess to a specialist who might not
    actually be the right fit.
    """
    category, confident = classify(message)
    if not confident:
        print(f"  [router] classified as '{category}' but low confidence — falling back to 'general'")
        category = "general"
    else:
        print(f"  [router] classified as '{category}' (confident)")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=HANDLER_SYSTEM_PROMPTS[category],
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": message}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return category, text


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Support router — categories: {CATEGORIES}. Type 'exit' to quit.\n")
    print("Try: \"I was charged twice this month\" (routes to billing)\n")

    while True:
        message = input("Message: ").strip()
        if message.lower() == "exit":
            print("Goodbye!")
            break
        if not message:
            continue

        category, response_text = route(message)
        print(f"\n[{category}] {response_text}\n")


if __name__ == "__main__":
    main()
