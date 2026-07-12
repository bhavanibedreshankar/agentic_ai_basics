"""
CONCEPT: Task Decomposition — breaking a complex, high-level goal into
smaller, well-scoped subtasks BEFORE any of them are executed. This
template is about the DECOMPOSITION step in isolation — the tree
structure a goal gets broken into, and what makes a decomposition good
or bad — not the execution loop that runs the pieces afterward.

That's the deliberate scope difference from
`../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`: that
template also decomposes a goal into steps, but immediately executes them
in a flat sequence and moves on — decomposition and execution are fused
into one pipeline there. Here, decomposition is its OWN artifact: a
`TaskNode` tree with a `decompose()` step you can inspect, validate, and
flatten independently of running anything. The tree can be MULTI-LEVEL
too (a subtask can itself be decomposed further if it's still too broad)
— `plan_and_execute`'s plan is always exactly one level of flat steps.

The other deliberate difference: this template's decomposition is
STRUCTURED OUTPUT (a JSON Schema with `subtasks: [...]`), not free text
parsed out afterward — the tree structure is guaranteed valid the moment
it comes back from the API, not hopefully-inferred from numbered list
text. One real constraint shows up here: Claude's structured outputs
don't support genuinely RECURSIVE schemas (a type referencing itself
arbitrarily deep), so the schema below hardcodes a fixed two-level shape
(goal -> subtasks -> optional sub-subtasks) instead of an unbounded tree
— a real recursive decomposition would need a different mechanism, e.g.
calling decompose() again on any subtask that still looks too broad.

Use case: decomposing a broad goal ("launch a podcast") into a task tree,
then flattening it into an execution-ready checklist. Type 'exit' to
quit.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

DECOMPOSE_SYSTEM_PROMPT = (
    "You decompose goals into subtasks. Given a goal, break it into 2-5 "
    "concrete subtasks. A subtask should be included as a leaf (no further "
    "subtasks) if it's small enough to execute directly in one step; if a "
    "subtask is still broad enough to warrant its own breakdown, give IT "
    "subtasks too. Don't go deeper than necessary — most goals need at "
    "most one or two levels."
)

# CONCEPT: structured output for a tree-SHAPED result, worked around a
# real limitation — Claude's structured outputs don't support a schema
# that references itself recursively, so an unbounded-depth tree isn't
# directly expressible in one call. This schema instead hardcodes exactly
# two levels: a subtask MAY have its own sub-subtasks, but a sub-subtask
# may not have further children. Good enough for most real goals (see the
# module docstring); a genuinely unbounded tree would call decompose()
# again on whichever subtask still looks too broad, rather than trying to
# get infinite depth out of one schema.
SUBTASK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, concrete description of this subtask"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A short, concrete description of this sub-subtask"},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            "description": "Further breakdown, only if this subtask is still too broad to execute directly. Empty array if this is already a leaf task.",
        },
    },
    "required": ["title", "subtasks"],
    "additionalProperties": False,
}

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The goal itself, restated concisely"},
        "subtasks": {"type": "array", "items": SUBTASK_SCHEMA, "description": "The top-level breakdown of the goal"},
    },
    "required": ["title", "subtasks"],
    "additionalProperties": False,
}


class TaskNode:
    """CONCEPT: the decomposition's actual shape — a tree, not a list.
    Built directly from the structured JSON response, so the tree
    structure is exactly what the model returned, not reconstructed by
    guessing at indentation in free text.
    """

    def __init__(self, title: str, subtasks: list["TaskNode"] | None = None):
        self.title = title
        self.subtasks = subtasks or []

    @classmethod
    def from_dict(cls, data: dict) -> "TaskNode":
        # The schema's two levels aren't shaped identically: a top-level
        # task and a subtask both carry a "subtasks" key, but a
        # sub-subtask (SUBTASK_SCHEMA's own "subtasks" items) is always a
        # leaf and carries only "title" — .get(..., []) handles that
        # missing key instead of a KeyError.
        children = [cls.from_dict(child) for child in data.get("subtasks", [])]
        return cls(data["title"], children)

    def is_leaf(self) -> bool:
        return not self.subtasks

    def print_tree(self, indent: int = 0) -> None:
        print("  " * indent + f"- {self.title}")
        for child in self.subtasks:
            child.print_tree(indent + 1)

    def leaves(self) -> list["TaskNode"]:
        """CONCEPT: flattening a multi-level tree into an execution-ready
        list. Only LEAVES are directly actionable — a node with children
        is a grouping label, not a task to run; recursing until there are
        no more children is what turns the tree back into something a
        run loop (like plan_and_execute's) could actually iterate over.
        """
        if self.is_leaf():
            return [self]
        flat: list[TaskNode] = []
        for child in self.subtasks:
            flat.extend(child.leaves())
        return flat


def decompose(goal: str) -> TaskNode:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=DECOMPOSE_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": TASK_SCHEMA}},
        messages=[{"role": "user", "content": goal}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return TaskNode.from_dict(json.loads(text))


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Task decomposition demo. Type 'exit' to quit.\n")
    print("Try: \"Launch a weekly podcast about local history.\"\n")

    while True:
        goal = input("Goal: ").strip()
        if goal.lower() == "exit":
            print("Goodbye!")
            break
        if not goal:
            continue

        root = decompose(goal)

        print("\n=== Task tree ===")
        root.print_tree()

        leaves = root.leaves()
        print(f"\n=== Flattened checklist ({len(leaves)} executable subtasks) ===")
        for i, leaf in enumerate(leaves, start=1):
            print(f"  {i}. {leaf.title}")
        print()


if __name__ == "__main__":
    main()
