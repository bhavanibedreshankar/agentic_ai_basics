"""
CONCEPT: Dynamic Prompt Construction — building a prompt's actual CONTENT at
runtime, from current variables and state, instead of sending one fixed,
hardcoded string every time.

../../01_Core_Architecture/system_prompt/system_prompt.py already shows
that the system prompt shapes behavior, but it does so by picking one
COMPLETE, pre-written prompt out of a small fixed set (`SYSTEM_PROMPTS`) —
every prompt in that dict is written in full, in advance, by a human. This
template does something structurally different: `build_system_prompt`
ASSEMBLES one prompt out of several independent, conditionally-included
SECTIONS, decided fresh on every call from whatever the current turn's real
state is — the user's account tier, whether their message reads as urgent,
which (if any) internal policies are actually relevant to what they asked,
and how many turns the conversation has gone on without resolving. No two
turns necessarily get the same system prompt, and nothing about which
sections appear is decided in advance.

This is also a different shape of "dynamic" than two other places in this
repo that use the word:
  - ../../17_LangChain/prompt_templates/prompt_templates.py fills SLOTS in
    one fixed template string ("Hi {name}, about your {product} ticket...")
    — the STRUCTURE never changes, only the values dropped into it. This
    template's structure itself changes: the urgency section, the policy
    section, and the escalation section each independently appear or don't,
    based on runtime conditions, not just fill blanks in a fixed shape.
  - ../../15_Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py
    also builds a system prompt at runtime, but by loading a JSON file of
    rules LEARNED FROM PAST feedback and persisting across process restarts.
    This template rebuilds its prompt from scratch every single call, purely
    from THIS turn's state — nothing here is learned or persisted.

Use case: a support assistant whose system prompt is assembled fresh each
turn from four independent pieces: a fixed persona, a tier-based service
block, an urgency block that only appears when the message reads as urgent,
and a relevant-policy block populated by a cheap keyword lookup (not a full
embeddings pipeline — see ../../09_RAG_and_Knowledge/rag/basic_rag.py for
that) against a tiny local policy set. Each constructed prompt is printed
before the response so the assembly is visible, not just its effect. Type
'exit' to end the conversation.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../01_Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the persona is the only FIXED piece — everything else below is
# assembled conditionally, on top of this, from runtime state.
# ---------------------------------------------------------------------------
BASE_PERSONA = (
    "You are a customer support assistant for a subscription software "
    "product. Be concise, direct, and specific — no filler pleasantries."
)

# ---------------------------------------------------------------------------
# CONCEPT: section 1 — tier-based instructions. Which block gets included
# depends on account state that's only known once a real user is talking,
# not something a human could write into one static prompt in advance.
# ---------------------------------------------------------------------------
def tier_instructions(tier: str) -> str:
    if tier == "premium":
        return (
            "This user is on the PREMIUM tier. Use a fast, white-glove "
            "tone, and proactively offer to escalate to a human if their "
            "issue isn't fully resolved in this message."
        )
    return (
        "This user is on the FREE tier. Offer best-effort help within "
        "standard policy; do not proactively offer refunds or extensions "
        "beyond what's explicitly documented."
    )


# ---------------------------------------------------------------------------
# CONCEPT: section 2 — an urgency block that only appears at all when a
# cheap keyword heuristic detects urgency in the CURRENT message. On a calm
# message, this section is simply absent from the prompt — not included and
# then told to be ignored, genuinely not part of the string sent to Claude.
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = {"urgent", "asap", "immediately", "furious", "angry", "right now", "escalate"}


def urgency_instructions(user_message: str) -> str:
    lowered = user_message.lower()
    if any(keyword in lowered for keyword in URGENCY_KEYWORDS):
        return (
            "The user's message reads as urgent or frustrated. Acknowledge "
            "that directly in your first sentence before addressing the "
            "issue, and prioritize speed over thoroughness."
        )
    return ""


# ---------------------------------------------------------------------------
# CONCEPT: section 3 — relevant policy facts, found by a deliberately cheap
# keyword lookup rather than the embedding-based retrieval used throughout
# ../../09_RAG_and_Knowledge/. This is the point: dynamic prompt
# construction doesn't require a full RAG stack — sometimes a handful of
# keyword checks is enough to decide what context belongs in the prompt.
# ---------------------------------------------------------------------------
POLICIES = {
    "refund": "Refunds are available within 30 days of purchase with proof of purchase.",
    "cancellation": "Subscriptions can be canceled anytime; access continues until the end of the current billing period.",
    "support-response-time": "Premium tier gets a 1-hour response target; free tier is best-effort within 24 hours.",
}
POLICY_KEYWORDS = {
    "refund": {"refund", "money back", "reimburse"},
    "cancellation": {"cancel", "subscription", "unsubscribe"},
    "support-response-time": {"response time", "how long", "priority", "premium support"},
}


def relevant_policies(user_message: str) -> list[str]:
    lowered = user_message.lower()
    matched = []
    for key, keywords in POLICY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            matched.append(POLICIES[key])
    return matched


def policy_instructions(user_message: str) -> str:
    matched = relevant_policies(user_message)
    if not matched:
        return ""
    facts = "\n".join(f"- {fact}" for fact in matched)
    return f"Relevant policy facts you can cite directly:\n{facts}"


# ---------------------------------------------------------------------------
# CONCEPT: section 4 — a block driven by CONVERSATION state (how many turns
# this session has gone), not the current message's content at all. This is
# what makes the assembly genuinely dynamic across a whole conversation,
# not just per-message: the exact same user message produces a different
# prompt on turn 1 than it would on turn 4.
# ---------------------------------------------------------------------------
STALL_THRESHOLD = 3


def stall_instructions(turn_count: int) -> str:
    if turn_count >= STALL_THRESHOLD:
        return (
            "This conversation has gone on for several turns without "
            "resolving. Proactively offer to connect the user with a "
            "human agent rather than continuing to troubleshoot alone."
        )
    return ""


# ---------------------------------------------------------------------------
# CONCEPT: the assembly itself. Each section is computed independently and
# only joined in if it produced non-empty text — this is the actual
# mechanic of "dynamic": the STRUCTURE of the final prompt (how many
# sections, which ones) varies call to call, not just the words inside a
# fixed template.
# ---------------------------------------------------------------------------
def build_sections(user_message: str, tier: str, turn_count: int) -> list[str]:
    """Returns only the sections that actually applied this turn — the
    single source of truth both `build_system_prompt` (for the API call)
    and `respond` (for printing what got included) build from.
    """
    sections = [
        BASE_PERSONA,
        tier_instructions(tier),
        urgency_instructions(user_message),
        policy_instructions(user_message),
        stall_instructions(turn_count),
    ]
    return [section for section in sections if section]


def build_system_prompt(user_message: str, tier: str, turn_count: int) -> str:
    return "\n\n".join(build_sections(user_message, tier, turn_count))


def respond(user_message: str, tier: str, turn_count: int) -> str:
    sections = build_sections(user_message, tier, turn_count)
    system_prompt = "\n\n".join(sections)
    print(f"  [constructed system prompt, {len(sections)} section(s) included]")
    print("  ---")
    for line in system_prompt.splitlines():
        print(f"  {line}")
    print("  ---")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Dynamic prompt construction demo — the system prompt is rebuilt from scratch every turn.")
    tier = input("Account tier ('free' or 'premium'): ").strip().lower()
    if tier not in ("free", "premium"):
        tier = "free"
    print(f"Using tier: {tier}. Type 'exit' to end the conversation.\n")
    print('Try: "How do I cancel my subscription?" then later "This is urgent, I need a refund ASAP!"\n')

    turn_count = 0
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        if not user_message:
            continue

        answer = respond(user_message, tier, turn_count)
        print(f"\nClaude: {answer}\n")
        turn_count += 1


if __name__ == "__main__":
    main()
