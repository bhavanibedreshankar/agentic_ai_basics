"""
CONCEPT: Checkpointing — saving a long-running task's intermediate state
to disk AS it progresses, so that if the process crashes, gets killed, or
is deliberately stopped partway through, restarting picks up from the
last completed step instead of starting the whole task over.

This is the one template in the repo where PERSISTENCE and RESUMPTION are
the whole point, not a side detail:
  - `../../Memory/episodic_memory/` and `../../Memory/semantic_memory/`
    persist DATA (facts, past interactions) — the conversation or task
    that produced that data is still expected to finish normally.
    Checkpointing persists TASK PROGRESS ITSELF — an in-flight,
    unfinished multi-step job — specifically so an interrupted run can
    continue exactly where it left off, not just remember things for
    next time.
  - `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`
    tracks step results in a plain Python list — gone the moment the
    process exits, deliberately (that template isn't about surviving a
    crash). This template takes that same "plan, then run steps in
    order accumulating results" shape and adds exactly one thing:
    writing progress to a checkpoint file after EVERY completed step, and
    reading it back at startup to skip whatever's already done.

The mechanic: before running the pipeline, load_checkpoint() checks for
an existing checkpoint file for this task_id. If one exists and lists
some steps as already completed, execution resumes from the first
INCOMPLETE step — completed steps are not re-run, and their SAVED
results are reused as context for the steps that still need to run.

Use case: a multi-step report-generation pipeline (outline -> section 1
-> section 2 -> section 3 -> conclusion) that can be interrupted (Ctrl-C,
or by choosing to quit mid-run) and resumed later without redoing
finished sections. Type 'exit' at the task prompt to quit; use Ctrl-C
mid-run to simulate a crash and test resumption on the next run.
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

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# CONCEPT: the fixed pipeline this template checkpoints through. Real
# checkpointing doesn't care whether the step sequence is fixed (like
# here) or dynamically planned (like
# ../task_decomposition/task_decomposition.py's tree, flattened) — the
# checkpointing MECHANISM is the same either way; a fixed list keeps this
# template's focus on save/resume rather than on planning.
STEPS = ["outline", "section_1", "section_2", "section_3", "conclusion"]

STEP_PROMPTS = {
    "outline": "Write a 3-4 bullet outline for a short report on the given topic.",
    "section_1": "Write the first section of the report, following the outline's first point.",
    "section_2": "Write the second section of the report, following the outline's second point.",
    "section_3": "Write the third section of the report, following the outline's third point.",
    "conclusion": "Write a brief conclusion that ties the sections together.",
}


def _checkpoint_path(task_id: str) -> Path:
    return CHECKPOINT_DIR / f"{task_id}.json"


def load_checkpoint(task_id: str) -> dict:
    """CONCEPT: resuming. If a checkpoint file exists, its `completed`
    dict maps step name -> that step's saved result. An empty/missing
    file means a fresh start — both cases return the same shape so the
    caller doesn't need to branch on "is this a resume or a fresh run".
    """
    path = _checkpoint_path(task_id)
    if not path.exists():
        return {"task_id": task_id, "completed": {}}
    return json.loads(path.read_text())


def save_checkpoint(task_id: str, completed: dict[str, str]) -> None:
    """CONCEPT: checkpointing itself. Called after EVERY step, not just
    at the end — if the process dies between step 2 and step 3, the
    checkpoint on disk reflects steps 1-2 as done, and a resume will
    correctly start at step 3, not step 1 or step 4.
    """
    path = _checkpoint_path(task_id)
    path.write_text(json.dumps({"task_id": task_id, "completed": completed}, indent=2))


def run_step(step: str, topic: str, completed_so_far: dict[str, str]) -> str:
    prior_context = "\n\n".join(f"{name}:\n{text}" for name, text in completed_so_far.items())
    prompt = f"Report topic: {topic}\n\nWork completed so far:\n{prior_context or '(nothing yet)'}\n\nNow: {STEP_PROMPTS[step]}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system="You are writing a short report, one section at a time.",
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def run_pipeline(task_id: str, topic: str) -> None:
    checkpoint = load_checkpoint(task_id)
    completed: dict[str, str] = checkpoint["completed"]

    already_done = [s for s in STEPS if s in completed]
    if already_done:
        print(f"  [resuming task '{task_id}' — {len(already_done)}/{len(STEPS)} steps already checkpointed: {already_done}]")
    else:
        print(f"  [starting task '{task_id}' fresh — no checkpoint found]")

    for step in STEPS:
        if step in completed:
            print(f"\n[{step}] already done (loaded from checkpoint) — skipping")
            continue

        print(f"\n[{step}] running...")
        result = run_step(step, topic, completed)
        print(result)

        # CONCEPT: save immediately after this step finishes — not
        # batched, not deferred to the end of the whole pipeline. The
        # checkpoint on disk is never more than one step behind reality.
        completed[step] = result
        save_checkpoint(task_id, completed)
        print(f"  [checkpointed: {step}]")

    print(f"\n=== Task '{task_id}' complete — all {len(STEPS)} steps done ===\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Checkpointed report pipeline. Type 'exit' at the task prompt to quit.")
    print("Interrupt with Ctrl-C mid-run to simulate a crash, then re-run with the same task id to resume.\n")

    while True:
        task_id = input("Task id (reuse the same id to resume a task): ").strip()
        if task_id.lower() == "exit":
            print("Goodbye!")
            break
        if not task_id:
            continue

        topic = input("Report topic (ignored when resuming past the outline step): ").strip()
        if not topic:
            topic = "(no topic given — infer from checkpointed context)"

        try:
            run_pipeline(task_id, topic)
        except KeyboardInterrupt:
            print(f"\n\n[interrupted — progress up to the last completed step is saved under task id '{task_id}']")
            print("Run again with the same task id to resume.\n")


if __name__ == "__main__":
    main()
