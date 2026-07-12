"""
CONCEPT: Chain-of-Thought (CoT) — prompting the model to reason step by
step BEFORE producing a final answer, instead of jumping straight to a
conclusion. For problems with multiple steps (arithmetic, logic, multi-
part questions), working through intermediate steps measurably improves
accuracy over answering directly — the model can catch its own errors
mid-reasoning that it would otherwise have committed to instantly.

This template shows THREE ways to get (or not get) that step-by-step
reasoning, so you can see the difference directly:

  1. DIRECT — no reasoning requested. Fast and cheap, but on multi-step
     problems this is where models are most likely to jump to a
     plausible-sounding wrong answer.
  2. PROMPTED CoT — the classic technique: explicitly instruct the model
     to "think step by step" and show its work as part of the visible
     response text, before stating a final answer. This is CoT the way
     it was originally popularized (Wei et al., 2022) — it works on any
     model, including ones with no built-in reasoning mode.
  3. NATIVE THINKING — Claude's built-in extended thinking
     (`thinking: {"type": "adaptive"}`). The model reasons in a separate
     `thinking` content block, distinct from its final answer text. This
     is the modern, API-native equivalent of prompted CoT — same idea
     (reason before answering), different mechanism (a dedicated block
     instead of visible prose), and Claude decides on its own how much
     reasoning a given question actually needs.

Type a question to see all three side by side. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def answer_direct(question: str) -> str:
    """No reasoning requested — answer immediately."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system="Answer with just the final answer. No explanation, no reasoning shown.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_with_cot_prompting(question: str) -> str:
    """CONCEPT: the classic prompted-CoT technique. The instruction below
    ("think step by step... then state your final answer") is the whole
    mechanism — no special API parameter, just an instruction that shapes
    what text the model produces. The reasoning ends up as regular,
    visible response text.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=(
            "Think through this step by step before answering. Show each "
            "step of your reasoning, then clearly state your final answer "
            "on its own line as 'Final answer: ...'."
        ),
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_with_native_thinking(question: str) -> tuple[str, str]:
    """CONCEPT: native extended thinking. `thinking={"type": "adaptive"}`
    tells Claude to reason in a dedicated `thinking` content block before
    producing its final text — the API separates the two explicitly,
    rather than the model mixing reasoning and answer together in one
    block of prose the way prompted CoT does. `display: "summarized"`
    asks for a readable summary of that reasoning (the default is to omit
    it from the response entirely, even though it still happened and was
    billed the same either way).

    Returns (thinking_text, answer_text) as a pair so the caller can show
    them separately.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question}],
    )
    thinking_text = "".join(block.thinking for block in response.content if block.type == "thinking")
    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return thinking_text, answer_text


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Chain-of-Thought comparison. Type 'exit' to quit.\n")
    print(
        "Try a multi-step problem, e.g.: \"A store has 15 apples. They "
        "sell 40% of them in the morning and 6 more in the afternoon. How "
        "many are left?\"\n"
    )

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        print("\n--- 1. Direct (no reasoning) ---")
        print(answer_direct(question))

        print("\n--- 2. Prompted chain-of-thought ---")
        print(answer_with_cot_prompting(question))

        print("\n--- 3. Native extended thinking ---")
        thinking_text, answer_text = answer_with_native_thinking(question)
        if thinking_text:
            print(f"[thinking]\n{thinking_text}\n")
        print(f"[answer]\n{answer_text}")
        print()


if __name__ == "__main__":
    main()
