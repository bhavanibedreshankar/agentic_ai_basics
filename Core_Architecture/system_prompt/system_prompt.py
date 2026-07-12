"""
CONCEPT: System Prompt — the fixed instruction set that defines an agent's
role, constraints, and (implicitly) how it's allowed to use whatever tools
it has. It's sent with every request, separate from the user's message, and
it's the main lever for shaping *how* an agent behaves without changing the
code that calls it.

../basics/basic.py already uses a SYSTEM_PROMPT constant, but only ever
shows the one it happens to be set to — you can't see what it's DOING
without something to contrast it against. This file makes the effect
explicit: run_with_prompt() is one fixed harness (an ordinary ask_claude
call), and main() sends the SAME user message through it under several
different system prompts. Every difference in the output — tone, length,
format, what's considered "in scope" to answer — comes from the system
prompt alone, since nothing else changed between the calls.
"""

import os
import sys

import anthropic

# --- API settings (see ../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"

# Three system prompts, each defining a different role AND a different
# constraint on the response — not just "be nicer" or "be terser", but
# genuinely different jobs for what is otherwise the identical model call.
SYSTEM_PROMPTS = {
    "terse_expert": (
        "You are a senior software engineer answering a colleague's quick "
        "question. Answer in at most two sentences. No hedging, no "
        "pleasantries, no repeating the question back."
    ),
    "eli5_tutor": (
        "You are a patient tutor explaining programming to a curious "
        "12-year-old who has never seen error messages before. Use a "
        "simple real-world analogy. Keep it encouraging and avoid jargon "
        "-- if you must use a technical term, define it immediately."
    ),
    "strict_json_api": (
        "You are a backend service, not a chat assistant. Respond with "
        "ONLY a single valid JSON object of the shape "
        '{\"summary\": string, \"likely_cause\": string} and nothing else '
        "-- no prose, no markdown fences, no explanation outside the JSON."
    ),
}

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def run_with_prompt(system_prompt: str, user_message: str) -> str:
    """One fixed call shape (see ../basics/basic.py's ask_claude) with the
    system prompt as a parameter instead of a module-level constant, so the
    same harness can be run under several different prompts in a row.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def compare_prompts(user_message: str, prompts: dict[str, str]) -> dict[str, str]:
    """Send the identical user_message through every named system prompt in
    `prompts`, returning each one's response for side-by-side comparison.
    """
    return {name: run_with_prompt(prompt, user_message) for name, prompt in prompts.items()}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    user_message = input(
        "Question to send under every system prompt (blank for a default demo):\n> "
    ).strip() or "Why does my Python code throw a KeyError when I access a dictionary?"

    print(f"\nSame question, {len(SYSTEM_PROMPTS)} different system prompts:\n  {user_message}\n")
    results = compare_prompts(user_message, SYSTEM_PROMPTS)

    for name, answer in results.items():
        print(f"--- {name} ---")
        print(answer)
        print()


if __name__ == "__main__":
    main()
