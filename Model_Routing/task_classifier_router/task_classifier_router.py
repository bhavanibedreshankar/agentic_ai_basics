"""
CONCEPT: Task-classifier model routing -- a single, cheap, upfront call
classifies how demanding an incoming request actually is, and THAT
classification picks which model tier answers it. Simple requests get a
fast/cheap model; only genuinely hard requests pay for the expensive one.

This is the same two-call SHAPE as
../../Agent_Frameworks_and_Patterns/router_agent/router_agent.py --
classify first via structured output, then dispatch -- but a different
AXIS of routing entirely:
  - router_agent.py classifies request DOMAIN (billing / technical /
    sales / general) and routes to a different SYSTEM PROMPT, always on
    the same model. Its own docstring even calls out that the classifier
    call is "a good candidate for a cheaper/faster model... but this
    template keeps everything on one model for consistency."
  - This template is that exact idea, followed through: classify request
    COMPLEXITY (simple / moderate / complex) and route to a different
    MODEL, on the theory that "what's the capital of France?" and "design
    a database schema migration plan with rollback strategy" don't
    deserve the same price tag.

The two axes compose in a real system: you could classify BOTH domain
and complexity and pick a (system_prompt, model) pair from a 2D grid --
this template only demonstrates the second axis, in isolation, to keep it
readable.

Use case: a general-purpose assistant fielding a mix of simple factual
lookups, moderate everyday writing/summarizing tasks, and complex
multi-step reasoning requests. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the routing table. Each complexity tier maps to a specific
# model -- cheapest/fastest for simple tasks, most capable for tasks that
# actually need it. Approximate per-million-token prices are included so
# the demo can print the (rough) cost difference the routing decision
# made -- check platform.claude.com/pricing for current rates.
# ---------------------------------------------------------------------------
TIERS = ["simple", "moderate", "complex"]

MODEL_FOR_TIER = {
    "simple": "claude-haiku-4-5-20251001",
    "moderate": "claude-sonnet-5",
    "complex": "claude-opus-4-8",
}

PRICE_PER_MILLION_INPUT = {
    "claude-haiku-4-5-20251001": 1.00,
    "claude-sonnet-5": 3.00,
    "claude-opus-4-8": 15.00,
}

CLASSIFY_SYSTEM_PROMPT = (
    "Classify how demanding the user's request is to answer well, into "
    "exactly one tier: 'simple' (a factual lookup or one-line answer, no "
    "real reasoning), 'moderate' (everyday writing, summarizing, or "
    "explaining something with a few moving parts), or 'complex' "
    "(multi-step reasoning, architecture/design decisions, anything "
    "where getting it wrong would be costly). When unsure between two "
    "tiers, pick the HIGHER one -- underestimating complexity risks a "
    "bad answer, overestimating only costs a few extra cents."
)

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"tier": {"type": "string", "enum": TIERS}},
    "required": ["tier"],
    "additionalProperties": False,
}


def classify_complexity(request: str) -> str:
    """CONCEPT: the router's entire job -- one structured-output call
    (same json_schema pattern as ../../Agent_Frameworks_and_Patterns/router_agent/router_agent.py's
    CLASSIFY_SCHEMA) returning a tier from a closed set, never free text
    to parse. Deliberately run on the CHEAPEST tier's model -- classifying
    "is this simple?" is itself a simple task, so paying Opus prices to
    make that judgment would defeat the entire point of routing.
    """
    response = client.messages.create(
        model=MODEL_FOR_TIER["simple"],
        max_tokens=64,
        system=CLASSIFY_SYSTEM_PROMPT,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": CLASSIFY_SCHEMA}},
        messages=[{"role": "user", "content": request}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)["tier"]


def route(request: str) -> tuple[str, str]:
    """Classify, then dispatch to the model that tier maps to. Returns
    (tier_used, response_text)."""
    tier = classify_complexity(request)
    model = MODEL_FOR_TIER[tier]
    print(f"  [router] classified as '{tier}' -> routing to {model}")

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system="You are a helpful assistant. Answer clearly and directly.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": request}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return tier, text


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Complexity-routing assistant. Type 'exit' to quit.")
    print('Try: "What year did WW2 end?" (simple) then')
    print('"Design a zero-downtime migration plan for a sharded Postgres database" (complex)\n')

    while True:
        request = input("You: ").strip()
        if request.lower() == "exit":
            print("Goodbye!")
            break
        if not request:
            continue
        tier, reply = route(request)
        print(f"\nClaude ({tier} -> {MODEL_FOR_TIER[tier]}): {reply}\n")


if __name__ == "__main__":
    main()
