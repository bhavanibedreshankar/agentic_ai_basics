"""
CONCEPT: Multi-model fallback -- routing driven purely by RELIABILITY,
not cost or task complexity. If the primary model is temporarily
unavailable (overloaded, rate-limited, a transient server error), retry
the SAME request against a different model instead of failing outright.

This is a different axis from every other template in this topic:
  - ../task_classifier_router/task_classifier_router.py and
    ../cost_aware_model_selection/cost_aware_model_selection.py route
    based on what the REQUEST needs (how hard it is, how much it should
    cost).
  - ../complexity_based_escalation/complexity_based_escalation.py routes
    based on what the FIRST MODEL'S ANSWER looked like (was it
    confident).
  - This template routes based on whether the call SUCCEEDED AT ALL. It
    doesn't care if the question was simple or complex, or what the
    answer said -- it only reacts to the primary model being unreachable
    right now, for reasons that have nothing to do with the request's
    content.

Only genuinely RETRYABLE errors trigger a fallback: server overload, rate
limiting, transient connection/timeout issues. Deliberately excluded are
errors like a bad request or an auth failure -- those would fail
IDENTICALLY against every model in the chain, so retrying elsewhere
wouldn't help and would just hide a real bug.

Use case: a chat assistant with a preferred primary model and a fallback
model to keep answering through transient outages. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and directly."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: the fallback chain, preferred model first. Every entry must be
# capable of handling the same request reasonably -- this isn't a quality
# ladder like ../complexity_based_escalation/complexity_based_escalation.py,
# just "something that can answer, in priority order."
FALLBACK_CHAIN = [
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
]

# CONCEPT: only these are worth failing over for. Each represents a
# transient condition that might not affect a DIFFERENT model or account
# for a request that will otherwise fail identically everywhere:
#   - OverloadedError / RateLimitError: this specific model is saturated
#     right now, not a problem with the request.
#   - InternalServerError / APIConnectionError / APITimeoutError:
#     transient infrastructure trouble, unrelated to the request's content.
# NotFoundError, BadRequestError, and AuthenticationError are deliberately
# NOT here -- a malformed request or bad credentials will fail the same
# way on every model, so retrying elsewhere would just mask the real bug.
RETRYABLE_ERRORS = (
    anthropic.OverloadedError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)


def ask_with_fallback(question: str) -> tuple[str, str]:
    """Try each model in FALLBACK_CHAIN in order, returning the first one
    that succeeds. Returns (answer, model_that_answered). Raises the last
    error if every model in the chain fails.
    """
    last_error: Exception | None = None
    for i, model in enumerate(FALLBACK_CHAIN):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": question}],
            )
        except RETRYABLE_ERRORS as error:
            last_error = error
            print(f"  [fallback] {model} unavailable ({type(error).__name__}) -- trying next model")
            continue

        if i > 0:
            print(f"  [fallback] recovered using {model}")
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, model

    raise RuntimeError(
        f"All {len(FALLBACK_CHAIN)} models in the fallback chain failed. Last error: {last_error}"
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Fallback assistant (chain: {' -> '.join(FALLBACK_CHAIN)}). Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        answer, model = ask_with_fallback(question)
        print(f"\nClaude ({model}): {answer}\n")


if __name__ == "__main__":
    main()
