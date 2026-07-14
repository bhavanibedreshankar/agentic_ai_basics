"""
CONCEPT: Selective context retention -- scoring each new piece of
information for importance THE MOMENT it arrives, and only letting the
important ones into the agent's persisted context at all. Everything
below the importance bar is discarded immediately and never stored
anywhere -- there is no "come back for it later."

This is a different technique from the other three context-shrinking
templates in this repo, even though all four fight the same problem
(context that grows without bound):
  - ../../Task_and_State_Management/context_management/pruning.py is
    REACTIVE: content goes into context first, and only gets removed
    later, once something newer has superseded it.
  - ../../Task_and_State_Management/context_management/retrieval.py keeps
    the full pool around in external storage forever and pulls relevant
    slices IN on demand -- nothing is ever truly discarded.
  - ../../Task_and_State_Management/context_management/summarization.py
    keeps a lossy compressed trace of EVERYTHING, including the
    unimportant parts, folded into a paragraph.
  - This template is PROACTIVE and genuinely destructive: low-importance
    input is scored and thrown away before it ever becomes part of the
    conversation or any store, on the theory that most of an incoming
    stream is noise that isn't worth even a compressed trace.

Use case: an on-call incident assistant ingesting a raw, noisy stream of
log lines during an outage. Heartbeats and routine INFO lines are noise;
error spikes, config changes, and deploys are signal. A cheap classification
call scores each line's importance 0-10 as it arrives; only lines scoring
at or above IMPORTANCE_THRESHOLD are folded into the incident's persistent
context. When you then ask the assistant questions, it answers ONLY from
retained lines -- asking about something that was discarded proves it's
genuinely gone, not just deprioritized.

Type 'exit' during the Q&A phase to end the session.
"""

from __future__ import annotations

import os
import re
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# Below this score (0-10), an incoming line is discarded outright.
IMPORTANCE_THRESHOLD = 6

# A simulated stream of raw log lines from an incident -- a deliberate mix
# of noise (heartbeats, routine info) and signal (errors, a config change,
# a deploy) to make the retain/discard split visible.
LOG_STREAM = [
    "INFO heartbeat: worker-3 alive, queue_depth=12",
    "INFO heartbeat: worker-1 alive, queue_depth=8",
    "ERROR checkout-service: payment gateway timeout after 30s, 214 requests affected",
    "INFO heartbeat: worker-2 alive, queue_depth=15",
    "DEBUG cache hit ratio 0.94 over last 60s",
    "WARN checkout-service: retrying payment gateway call (attempt 2/3)",
    "INFO heartbeat: worker-3 alive, queue_depth=13",
    "DEPLOY: checkout-service rolled back from v2.14.1 to v2.14.0 at 14:32 UTC",
    "INFO heartbeat: worker-1 alive, queue_depth=9",
    "CONFIG_CHANGE: payment gateway timeout raised from 10s to 30s by @jamie",
    "DEBUG gc pause 12ms",
    "INFO heartbeat: worker-2 alive, queue_depth=14",
]


def score_importance(line: str) -> int:
    """CONCEPT: the retention gate. A small, cheap classification call --
    low max_tokens, a narrow prompt -- whose only job is judging whether
    ONE line is worth keeping. This is intentionally the kind of small,
    well-defined task that ../../Model_Routing/task_classifier_router/task_classifier_router.py
    would route to a cheap/fast model rather than the main assistant
    model, since the two concerns (classify importance, then answer
    questions) don't need the same model tier.

    The model's raw text output is untrusted input, same as any tool
    result -- it's parsed defensively with a regex and clamped to the
    valid range rather than trusted as a well-formed integer.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=8,
        output_config={"effort": "low"},
        system=(
            "You score how important a single log line is for an on-call "
            "engineer trying to understand an ongoing incident, on a 0-10 "
            "scale. Routine heartbeats, debug noise, and healthy metrics "
            "score low (0-2). Errors, warnings, config changes, and deploys "
            "score high (7-10). Respond with ONLY the integer, nothing else."
        ),
        messages=[{"role": "user", "content": line}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    match = re.search(r"\d+", text)
    if not match:
        return 0  # fail closed: unparseable score is treated as unimportant, not important
    return max(0, min(10, int(match.group())))


def ingest(line: str, retained_context: list[str], discarded_log: list[str]) -> None:
    """Score one incoming line and either fold it into retained_context or
    drop it. Discarded lines are appended to discarded_log ONLY so this
    demo can show you what was thrown away -- a real agent wouldn't keep
    that list at all, since the whole point is not paying to store it.
    """
    score = score_importance(line)
    if score >= IMPORTANCE_THRESHOLD:
        retained_context.append(line)
        print(f"  [retained  score={score:2d}] {line}")
    else:
        discarded_log.append(line)
        print(f"  [discarded score={score:2d}] {line}")


def build_system_prompt(retained_context: list[str]) -> str:
    """CONCEPT: only retained lines ever reach the assistant's context.
    Contrast with ../../Memory/memory_management/basic_agentic_memory.py,
    which injects EVERY saved fact unconditionally -- here, most of the
    raw stream never gets this far in the first place."""
    if not retained_context:
        notes = "(no incident facts have been retained yet)"
    else:
        notes = "\n".join(f"- {line}" for line in retained_context)
    return (
        "You are an on-call incident assistant. Answer questions using ONLY "
        "the retained incident facts below. If something isn't in this list, "
        "say you don't have that information -- do not guess or invent details.\n\n"
        f"Retained incident facts:\n{notes}"
    )


def ask(question: str, retained_context: list[str]) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(retained_context),
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Ingesting a simulated incident log stream...\n")
    retained_context: list[str] = []
    discarded_log: list[str] = []
    for line in LOG_STREAM:
        ingest(line, retained_context, discarded_log)

    print(
        f"\nDone: {len(retained_context)} retained / {len(discarded_log)} discarded "
        f"out of {len(LOG_STREAM)} lines.\n"
    )
    print("Ask questions about the incident. Type 'exit' to end.")
    print('Try: "What changed right before the incident?" then "What was worker-2\'s queue depth?"\n')

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print(f"\nClaude: {ask(question, retained_context)}\n")


if __name__ == "__main__":
    main()
