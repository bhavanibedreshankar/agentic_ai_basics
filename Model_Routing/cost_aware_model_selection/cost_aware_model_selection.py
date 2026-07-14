"""
CONCEPT: Cost-aware model selection -- picking a model tier using FREE,
local heuristics computed from the request itself, with no extra API
call at all, and tracking a running session budget that forces cheaper
tiers once spending gets close to a cap.

This is deliberately a different mechanism from
../task_classifier_router/task_classifier_router.py, which spends a real
(if cheap) API call to have Claude itself judge complexity. That
classification call is more accurate, but it isn't free -- every request
pays for two calls instead of one. This template's heuristic scorer costs
nothing to run: it looks at surface features of the request text (length,
presence of code, multi-part questions, "explain in detail"-style
phrasing) and scores complexity locally, in Python, before any API call
happens. The trade-off is bluntness -- a heuristic can be fooled by a
short-but-hard question, or a long-but-easy one -- for zero marginal cost.

The second half of "cost-aware" here is a SESSION BUDGET: once cumulative
estimated spend for the session crosses BUDGET_CAP_USD, every subsequent
request is forced to the cheapest tier regardless of what the heuristic
says, on the theory that staying within budget matters more than getting
the ideal tier for one more request.

Use case: a customer-facing chat widget with a fixed per-session cost
budget -- common for a free tier or a cost-capped internal tool. Type
'exit' to end the conversation and see the spending summary.
"""

from __future__ import annotations

import os
import re
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

MODEL_FOR_TIER = {
    "cheap": "claude-haiku-4-5-20251001",
    "standard": "claude-sonnet-5",
}

# Approximate per-million-token input prices -- check platform.claude.com/pricing
# for current rates; only used here to make the budget concept concrete.
PRICE_PER_MILLION_INPUT = {
    "claude-haiku-4-5-20251001": 1.00,
    "claude-sonnet-5": 3.00,
}
PRICE_PER_MILLION_OUTPUT = {
    "claude-haiku-4-5-20251001": 5.00,
    "claude-sonnet-5": 15.00,
}

# Once cumulative estimated session spend reaches this, force the
# cheapest tier for every remaining request in the session.
BUDGET_CAP_USD = 0.02

# Heuristic signals that a request needs more than a cheap model's worth
# of reasoning. None of this is an API call -- it's plain string checks.
COMPLEXITY_KEYWORDS = (
    "explain in detail", "step by step", "design", "architecture", "trade-off",
    "compare and contrast", "why does", "root cause", "debug", "optimize",
)
LONG_REQUEST_CHARS = 240  # requests longer than this lean toward 'standard'


def score_complexity(request: str) -> str:
    """CONCEPT: the free heuristic. No API call -- just surface features
    of the text. Any one signal (length, a complexity keyword, multiple
    question marks suggesting a multi-part ask) is enough to bump the
    request up to 'standard'; otherwise it stays 'cheap'.
    """
    text = request.lower()
    if len(request) > LONG_REQUEST_CHARS:
        return "standard"
    if any(keyword in text for keyword in COMPLEXITY_KEYWORDS):
        return "standard"
    if request.count("?") > 1:
        return "standard"
    return "cheap"


class SessionBudget:
    """CONCEPT: a running cost tracker that can override the heuristic's
    tier choice. `choose_tier` is where the two halves of "cost-aware"
    meet: ask the free heuristic what it WOULD pick, then downgrade to
    the cheapest tier anyway if the budget is already spent, regardless
    of what the heuristic wanted.
    """

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    def choose_tier(self, request: str) -> str:
        heuristic_tier = score_complexity(request)
        if self.spent_usd >= self.cap_usd and heuristic_tier != "cheap":
            print(
                f"  [budget] ${self.spent_usd:.4f} already spent (cap ${self.cap_usd:.4f}) "
                f"-- forcing 'cheap' instead of '{heuristic_tier}'"
            )
            return "cheap"
        return heuristic_tier

    def record(self, model: str, usage) -> None:
        cost = (
            usage.input_tokens * PRICE_PER_MILLION_INPUT[model]
            + usage.output_tokens * PRICE_PER_MILLION_OUTPUT[model]
        ) / 1_000_000
        self.spent_usd += cost

    def summary(self) -> str:
        return f"Estimated session spend: ${self.spent_usd:.4f} (cap: ${self.cap_usd:.4f})"


def ask(request: str, budget: SessionBudget) -> tuple[str, str]:
    tier = budget.choose_tier(request)
    model = MODEL_FOR_TIER[tier]
    print(f"  [router] heuristic tier='{tier}' -> {model}")

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system="You are a helpful assistant. Answer clearly and directly.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": request}],
    )
    budget.record(model, response.usage)
    text = "".join(block.text for block in response.content if block.type == "text")
    return tier, text


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Budget-capped assistant (cap: ${BUDGET_CAP_USD:.4f}/session). Type 'exit' to end.")
    print('Try a few short questions, then one with "design" or "step by step" in it,')
    print("and keep going until the budget forces everything to the cheap tier.\n")

    budget = SessionBudget(BUDGET_CAP_USD)
    while True:
        request = input("You: ").strip()
        if request.lower() == "exit":
            print(f"\n--- {budget.summary()} ---")
            break
        if not request:
            continue
        tier, reply = ask(request, budget)
        print(f"\nClaude ({tier}): {reply}\n")


if __name__ == "__main__":
    main()
