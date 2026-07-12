"""
CONCEPT: Self-Evolving Agents — AI systems that autonomously update their
own capabilities (here: the system prompt that drives their behavior)
through continuous, closed-loop feedback, with the update PERSISTING
across runs rather than staying inside one conversation.

This is a different axis than every other "the agent gets better" template
in the repo:
  - ../../Planning_and_Reasoning/self_reflection/self_reflection.py revises
    a single OUTPUT (one email draft) through critique-and-revise, entirely
    in memory. Once main() returns, nothing learned survives — ask it the
    same kind of question tomorrow and it starts from zero again.
  - ../../Memory/episodic_memory/episodic_memory.py persists a LOG of past
    interactions and lets the agent choose to recall_episodes and imitate
    what worked before — but the underlying instructions never change; the
    agent re-derives the lesson from raw history on every single run.
  - Here, feedback on an answer is distilled ONCE into a short, reusable
    RULE, and that rule is spliced directly into SYSTEM_PROMPT for every
    future call — including calls in a brand-new process, since the rules
    live in evolved_rules.json, not in this conversation's message list.
    The agent's own capability (what it knows to do without being told)
    grows; ../../Core_Architecture/system_prompt/system_prompt.py's prompt,
    by contrast, is fixed at write time and never changes itself.

The loop that makes this "self-evolving" rather than just "logged":
  answer -> user feedback -> (if negative) an LLM call turns that feedback
  into ONE generalizable rule -> the rule is appended to persistent state
  -> the VERY NEXT answer is generated with that rule already in its
  system prompt. Closing the loop back into the same prompt that produced
  the mistake is the whole mechanic; everything else in this file exists
  to make that one step safe and observable.

Self-modification without a limit is a real risk (a bad rule could
compound, or the prompt could grow without bound) — MAX_RULES caps how
much the agent can rewrite itself in one run, the same spirit as the caps
in ../../Safety_and_Control/guardrails.py, just applied to the agent's own
instructions instead of its actions.

Demonstrated on a coding-help assistant that tends to give answers missing
a detail the user cares about (e.g. Python version, error handling) —
rate an answer thumbs-down with a reason, and watch the next answer to a
similar question already account for it. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

BASE_SYSTEM_PROMPT = (
    "You are a concise coding assistant. Answer the user's programming "
    "question directly, with a short code example when useful."
)

# CONCEPT: a hard cap on self-modification — without this, a noisy or
# adversarial feedback stream could let the agent rewrite its own
# instructions without bound. Once the cap is hit, feedback is still
# accepted from the user, but no new rule gets added (see evolve()).
MAX_RULES = 8

RULES_FILE = Path(__file__).parent / "evolved_rules.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "rule": {
            "type": "string",
            "description": (
                "One short, general, imperative instruction (not tied to "
                "this specific question) that would have prevented the "
                "feedback's complaint. E.g. 'Always mention which Python "
                "version an example targets.'"
            ),
        }
    },
    "required": ["rule"],
    "additionalProperties": False,
}

RULE_PROPOSER_SYSTEM_PROMPT = (
    "You turn one piece of negative feedback about an AI assistant's "
    "answer into a single, general, reusable instruction that would "
    "prevent that specific complaint from recurring on FUTURE, possibly "
    "different, questions. Keep it to one sentence, imperative mood, and "
    "generalized — not a restatement of this one answer's mistake."
)


def _load_rules() -> list[str]:
    if not RULES_FILE.exists():
        return []
    return json.loads(RULES_FILE.read_text())


def _save_rules(rules: list[str]) -> None:
    RULES_FILE.write_text(json.dumps(rules, indent=2))


def build_system_prompt(rules: list[str]) -> str:
    """CONCEPT: the agent's live instructions are BASE_SYSTEM_PROMPT plus
    everything it has learned so far — this is called fresh before every
    answer, so a rule added mid-session (or in a past process entirely)
    is already in effect for the very next call.
    """
    if not rules:
        return BASE_SYSTEM_PROMPT
    learned = "\n".join(f"- {rule}" for rule in rules)
    return f"{BASE_SYSTEM_PROMPT}\n\nRules learned from past feedback:\n{learned}"


def answer(question: str, rules: list[str]) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(rules),
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def propose_rule(question: str, response_text: str, feedback: str) -> str:
    """CONCEPT: feedback on one answer is distilled into a rule general
    enough to help on DIFFERENT future questions. Structured output (same
    {schema} + json.loads pattern as
    ../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py)
    keeps this a clean single string instead of free text you'd have to
    parse a rule out of.
    """
    prompt = (
        f"Question: {question}\n\nAssistant's answer:\n{response_text}\n\n"
        f"User's negative feedback: {feedback}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=RULE_PROPOSER_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": RULE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)["rule"]


def evolve(question: str, response_text: str, feedback: str, rules: list[str]) -> list[str]:
    """Apply one round of the closed feedback loop, returning the
    (possibly updated) rule set. Kept separate from propose_rule so the
    cap/dedupe policy is testable without hitting the API.
    """
    if len(rules) >= MAX_RULES:
        print(f"(MAX_RULES={MAX_RULES} reached — feedback noted, but no new rule added)")
        return rules

    new_rule = propose_rule(question, response_text, feedback)
    if new_rule in rules:
        print("(equivalent rule already learned — nothing new to add)")
        return rules

    rules = [*rules, new_rule]
    _save_rules(rules)
    print(f"[agent evolved] added rule: {new_rule}")
    return rules


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    rules = _load_rules()
    print("Self-evolving coding assistant. Type 'exit' to quit.")
    print(f"Starting with {len(rules)} previously learned rule(s).\n")
    print("Try: \"How do I read a file in Python?\", then rate it down with a reason like")
    print("\"didn't say what to do if the file is missing\", then ask a similar question again.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        response_text = answer(question, rules)
        print(f"\n{response_text}\n")

        rating = input("Rate this answer (up/down, or press enter to skip): ").strip().lower()
        if rating == "down":
            feedback = input("What was wrong with it? ").strip()
            if feedback:
                rules = evolve(question, response_text, feedback, rules)
        print()


if __name__ == "__main__":
    main()
