"""
CONCEPT: The agentic conversation loop — how a "chat" is actually built on
top of a stateless API.

Claude's API has no memory between calls. Every request is independent, so
to have a multi-turn conversation, YOUR code must keep track of everything
said so far and resend the full history on every request. This script is a
minimal example of that pattern: a `while True` loop that keeps prompting
the user, growing a `messages` list, and sending the whole thing back each
time until the user types "exit".
"""

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = "You are a helpful assistant. Answer questions on any topic clearly and concisely."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def ask_claude(messages: list[dict]) -> str:
    """Send the full conversation history to Claude and return its reply.

    Unlike ../../basics/basic.py's ask_claude (which took a single string), this takes the
    entire `messages` list built up so far. Claude only "remembers" what's in
    this list — nothing more, nothing less.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Chat with Claude. Type 'exit' to end the conversation.\n")

    # This list IS the conversation. Each entry is one turn: a dict with a
    # "role" (who spoke) and "content" (what was said). It starts empty and
    # grows by two entries (one user, one assistant) every loop iteration.
    messages: list[dict] = []

    # ---- THE AGENTIC LOOP ----
    # This is the pattern behind every conversational agent: keep looping,
    # get input, call the model with the accumulated state, act on the
    # result (here, just print it), and repeat — until an exit condition.
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        # 1. Record what the user said.
        messages.append({"role": "user", "content": user_input})

        # 2. Ask Claude, passing the ENTIRE history so far — not just the
        #    latest message. This is what makes the conversation coherent
        #    across turns (Claude can refer back to earlier messages).
        reply = ask_claude(messages)

        # 3. Record Claude's reply too, so the NEXT loop iteration includes
        #    it in the history. Forgetting this step is a common bug — the
        #    conversation would "forget" everything Claude said.
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
