"""
CONCEPT: Procedural memory — an agent's own INSTRUCTIONS (its system
prompt) improving over time from feedback on its performance, with the
improved prompt persisted so every future run starts from the upgraded
version, not the original.

../../Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py builds this exact
mechanic by hand: negative feedback on an answer gets sent to a SEPARATE
LLM call that distills it into one short, reusable RULE, and that rule is
spliced verbatim into `SYSTEM_PROMPT` and appended to `evolved_rules.json`.
This template is the LangMem-native version of the same closed loop —
`create_prompt_optimizer(kind="prompt_memory")` takes the CURRENT prompt
plus a trajectory (the conversation) annotated with feedback, and returns
a REWRITTEN prompt, deciding on its own how to incorporate the lesson
rather than appending a discrete rule to a growing list. That's the real
difference in what each version controls:
  - `self_evolving_agents.py`: the rule-distillation and splicing are both
    explicit, inspectable steps written into this repo — you can read
    exactly how a rule gets worded and exactly where it lands in the
    prompt.
  - This file: the REWRITE itself is the optimizer's judgment call — it
    might reword existing instructions, add a new sentence, or restructure
    the whole prompt, and you only see the before/after, not the
    reasoning steps in between (unless you inspect the optimizer's
    internal `analysis`/`logic` output, which this template does print,
    since it's the closest thing to the rule).

Use case: a support assistant whose prompt evolves in place, persisted to
`evolved_prompt.json` (gitignored, same role as `evolved_rules.json` in
../../Self_Evolving_Agents/). Give it feedback after an answer and the
prompt updates for the very next turn, in this run AND in a fresh
process. Type 'exit' to end the session.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import langmem
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langmem.prompts.types import AnnotatedTrajectory

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512

DEFAULT_PROMPT = "You are a helpful support assistant."
PROMPT_FILE = Path(__file__).parent / "evolved_prompt.json"


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        return DEFAULT_PROMPT
    return json.loads(PROMPT_FILE.read_text())["prompt"]


def save_prompt(prompt: str) -> None:
    PROMPT_FILE.write_text(json.dumps({"prompt": prompt}, indent=2))


def build_optimizer(llm: BaseChatModel) -> Runnable:
    # CONCEPT: kind="prompt_memory" is LangMem's simplest optimizer — one
    # model call that reads the current prompt plus feedback and decides
    # whether and how to rewrite it. ("gradient" and "metaprompt" run a
    # multi-step reflect-then-revise process internally for harder cases;
    # not needed to demonstrate the mechanic here.)
    return langmem.create_prompt_optimizer(llm, kind="prompt_memory")


def improve_from_feedback(optimizer: Runnable, current_prompt: str, user_message: str, assistant_reply: str, feedback: str) -> str:
    trajectory = AnnotatedTrajectory(
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_reply},
        ],
        feedback=feedback,
    )
    return optimizer.invoke({"trajectories": [trajectory], "prompt": current_prompt})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    optimizer = build_optimizer(llm)

    print("Self-improving support assistant (procedural memory demo). Type 'exit' to quit.")
    print(f"Current prompt: {load_prompt()!r}\n")
    print("After each reply, you can give feedback (or press enter to skip) — negative feedback rewrites the prompt.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        if not user_message:
            continue

        current_prompt = load_prompt()
        reply = llm.invoke([{"role": "system", "content": current_prompt}, {"role": "user", "content": user_message}])
        print(f"Claude: {reply.content}\n")

        feedback = input("Feedback on that reply (blank to skip): ").strip()
        if feedback:
            new_prompt = improve_from_feedback(optimizer, current_prompt, user_message, reply.content, feedback)
            if new_prompt and new_prompt != current_prompt:
                save_prompt(new_prompt)
                print(f"\n[prompt updated]\n  before: {current_prompt!r}\n  after:  {new_prompt!r}\n")
            else:
                print("\n[optimizer left the prompt unchanged]\n")


if __name__ == "__main__":
    main()
