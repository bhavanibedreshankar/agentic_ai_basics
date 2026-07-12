"""
CONCEPT: The basics of calling the Claude API — auth, model settings, and a
single request/response call. This is the smallest possible building block:
no conversation memory, no tools, just "send a message, get an answer."

Everything in the later templates (../Execution_Loops/agentic_loop/basic_agentic_loop.py, ../Tools_and_Actions/tool_use/basic_agentic_tools.py)
builds on this same call shape.
"""

import os
import sys

import anthropic

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
# MODEL: which Claude model handles the request. Different models trade off
# speed/cost against intelligence — Sonnet is a strong default for most tasks.
MODEL = "claude-sonnet-5"

# MAX_TOKENS: the hard ceiling on how many tokens Claude is allowed to generate
# in its response. If a response would be longer than this, it gets cut off.
MAX_TOKENS = 16000

# EFFORT: how much the model "thinks" before answering. Lower = faster/cheaper,
# higher = more thorough. Options: low, medium, high, xhigh, max.
EFFORT = "medium"

# SYSTEM_PROMPT: standing instructions sent with every request that shape
# *how* Claude should behave — separate from the user's actual message.
SYSTEM_PROMPT = (
    "You are a Python coding assistant. Given a description of what the user "
    "wants, respond with only the Python code that accomplishes it, in a single "
    "fenced code block. Add brief inline comments only where the logic isn't "
    "obvious."
)

# The client object handles authentication and the HTTP calls to the API.
# Anthropic() with no arguments looks for ANTHROPIC_API_KEY in your environment
# automatically — you never hardcode a key in the source.
client = anthropic.Anthropic()


def ask_claude(message: str) -> str:
    """Send a single message to Claude and return its text response.

    This is the core pattern for every Claude API call: build a request with
    a model, a token limit, optional system instructions, and a list of
    messages — then read the response back out.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        # `messages` is always a list, even for a single turn. Each entry has
        # a role ("user" or "assistant") and content. Here we only send one
        # user turn — there's no memory of past turns in this script.
        messages=[{"role": "user", "content": message}],
    )

    # response.content is a LIST of content blocks, not a plain string.
    # A response can contain multiple block types (text, tool calls, etc.),
    # so we filter for "text" blocks and join them into one string.
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    # Fail fast with a clear message if the API key isn't set, rather than
    # letting the SDK raise a less obvious authentication error later.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    request = input("What Python code would you like me to generate?\n> ").strip()
    if not request:
        print("No request given.", file=sys.stderr)
        sys.exit(1)

    print("\nGenerating...\n")
    print(ask_claude(request))


if __name__ == "__main__":
    main()
