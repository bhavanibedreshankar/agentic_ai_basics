"""
CONCEPT: Swarm — a loosely coordinated collection of agents working in
PARALLEL, each independently, often with no central orchestrator
deciding who does what or in what order.

Contrast with ../orchestrator/: there, ONE agent decides which
specialist to call and in what sequence — deliberate, sequential
delegation, always in control. A swarm has no such coordinator: every
agent gets the SAME input at the SAME time, works completely
independently (no agent knows the others exist or sees their output),
and the results are only combined AFTERWARD, in a separate synthesis
step that isn't part of the swarm itself.

Contrast with ../../Planning_and_Reasoning/tree_of_thought/: ToT runs
several attempts at the SAME narrow problem and picks the SINGLE best
one, discarding the rest. A swarm runs DIFFERENT SPECIALIZED PERSPECTIVES
on a BROADER problem and MERGES all of them together — nothing is
discarded, because each member is answering a different question, not
competing to answer the same one.

This is also the first template in the whole repo to make genuinely
CONCURRENT API calls. Every other multi-call template
(prompt_chaining, plan_and_execute, orchestrator, supervisor_pattern)
calls Claude sequentially — one request waiting for the previous one to
finish. concurrent.futures.ThreadPoolExecutor here fires every swarm
member's call at once; the demo prints each member's completion time so
the speedup versus calling them one after another is directly visible.

Use case: an investment proposal swarm — three independent analysts
(financial, risk, market) evaluate the same proposal with no awareness
of each other; a synthesis step then combines all three perspectives
into one recommendation. Type 'exit' to quit.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: the swarm members. Each has its own narrow persona and no
# awareness that the others exist — nothing here references another
# analyst's role or expects to see their output.
ANALYSTS = {
    "financial": "You are a financial analyst. Evaluate the proposal purely on financial merit: costs, revenue potential, ROI. 3-4 sentences.",
    "risk": "You are a risk analyst. Evaluate the proposal purely on risk: what could go wrong, regulatory concerns, downside scenarios. 3-4 sentences.",
    "market": "You are a market analyst. Evaluate the proposal purely on market fit: competition, timing, customer demand. 3-4 sentences.",
}


def run_analyst(name: str, system_prompt: str, proposal: str) -> tuple[str, str, float]:
    """One swarm member's independent evaluation. Returns (name, text,
    elapsed_seconds) — the timing is purely to make the parallelism
    visible in the demo output, not part of the pattern itself.
    """
    start = time.monotonic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": proposal}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    elapsed = time.monotonic() - start
    return name, text, elapsed


def run_swarm(proposal: str) -> dict[str, str]:
    """CONCEPT: fan out to every swarm member AT ONCE via a thread pool —
    no coordinator decides an order, because there is no order; every
    member starts at (approximately) the same moment. as_completed yields
    results in whatever order they actually finish, not submission order.
    """
    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ANALYSTS)) as executor:
        futures = [
            executor.submit(run_analyst, name, system_prompt, proposal)
            for name, system_prompt in ANALYSTS.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            name, text, elapsed = future.result()
            results[name] = text
            print(f"  [{name} analyst done in {elapsed:.1f}s]")
    return results


def synthesize(proposal: str, results: dict[str, str]) -> str:
    """A separate step, distinct from the swarm itself — the swarm
    produces independent perspectives; synthesis is where they finally
    get combined into one answer.
    """
    combined = "\n\n".join(f"{name.title()} analyst:\n{text}" for name, text in results.items())
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Synthesize the following independent analyses into one balanced recommendation.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": f"Proposal: {proposal}\n\n{combined}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Investment proposal swarm — {len(ANALYSTS)} analysts in parallel. Type 'exit' to quit.\n")
    print("Try: \"We're proposing to launch a subscription box for artisanal coffee.\"\n")

    while True:
        proposal = input("Proposal: ").strip()
        if proposal.lower() == "exit":
            print("Goodbye!")
            break
        if not proposal:
            continue

        swarm_start = time.monotonic()
        results = run_swarm(proposal)
        swarm_elapsed = time.monotonic() - swarm_start
        print(f"  [swarm total wall-clock time: {swarm_elapsed:.1f}s — compare to the sum of individual times above]")

        for name, text in results.items():
            print(f"\n--- {name.title()} analyst ---\n{text}")

        recommendation = synthesize(proposal, results)
        print(f"\n=== Synthesized recommendation ===\n{recommendation}\n")


if __name__ == "__main__":
    main()
