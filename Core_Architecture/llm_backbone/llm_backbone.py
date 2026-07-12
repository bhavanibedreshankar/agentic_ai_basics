"""
CONCEPT: LLM (Backbone) — the large language model that serves as the
reasoning engine inside an agent. The agent's *architecture* (its system
prompt, its tools, its loop) is separate from *which model* powers the
reasoning at each step — the backbone is a swappable component, not
something baked into the rest of the code.

This file proves that by literally swapping it: run_with_backbone() is one
fixed harness — same system prompt, same tool, same loop shape as
../tool_use/basic_agentic_tools.py — with `model` as a plain parameter.
main() calls it twice, on the IDENTICAL task, passing two different model
IDs. Nothing else about the harness changes. What differs between the two
runs (answer quality, tone, tokens used, latency) is entirely attributable
to the backbone, because everything else was held constant — the same idea
../basics/basic.py's MODEL constant hints at, made explicit and comparable
here instead of fixed once at the top of the file.
"""

from __future__ import annotations

import os
import sys
import time

import anthropic

# --- API settings (see ../basics/basic.py for what each of these means) ---
# Note there's no single MODEL constant here, unlike every other template in
# this repo — that's the point. The model is a parameter to run_with_backbone(),
# not a fixed setting, because this file's whole subject is swapping it.
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the 'calculate' tool for any arithmetic "
    "instead of computing it yourself. Give a concise final answer."
)

# Two backbones to compare on the same task. Both are Claude models here
# (this repo only calls the Claude API — see ../basics/README.md), but the
# harness below would work identically with any model string the API
# accepts; the LLM is just the thing plugged into the `model` parameter.
BACKBONES = ["claude-haiku-4-5", "claude-sonnet-5"]

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 + 2 * 3'"},
            },
            "required": ["expression"],
        },
    },
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def _calculate(expression: str) -> str:
    # Deliberately permissive (not the ast-based safe eval used elsewhere,
    # e.g. ../agent/agent.py) to keep this file focused on the backbone-swap
    # concept rather than re-demonstrating safe evaluation.
    allowed = set("0123456789.+-*/() ")
    if not set(expression) <= allowed:
        raise ValueError(f"Unsupported expression: {expression}")
    return str(eval(expression))  # noqa: S307 - character-restricted above


def run_with_backbone(model: str, task: str) -> dict:
    """Run one fixed harness to completion, with `model` as the only thing
    that changes between calls. Returns the final answer plus stats that
    make the backbone's effect on the outcome visible.

    This loop's SHAPE is identical to ../tool_use/basic_agentic_tools.py's
    run_turn() — the only structural difference is that `model` comes in as
    an argument instead of being read from a module-level constant.
    """
    messages: list[dict] = [{"role": "user", "content": task}]
    start = time.monotonic()
    total_input_tokens = 0
    total_output_tokens = 0

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        # CONCEPT: usage is reported per backbone call — see
        # ../token_tracking/basic_token_tracking.py for a dedicated deep
        # dive into this same `response.usage` object.
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return {
                "model": model,
                "answer": final_text,
                "seconds": round(time.monotonic() - start, 2),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result_text = _calculate(block.input["expression"])
                    is_error = False
                except Exception as exc:  # noqa: BLE001 - surface to the model
                    result_text, is_error = f"Error: {exc}", True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )
        messages.append({"role": "user", "content": tool_results})


def compare_backbones(task: str, models: list[str]) -> list[dict]:
    """Run the identical harness against every backbone in `models`, on the
    identical task, and return each run's results for side-by-side
    comparison.
    """
    return [run_with_backbone(model, task) for model in models]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    task = input("Task to give both backbones (blank for a default demo):\n> ").strip() or (
        "A store is offering 30% off a $85 jacket. What's the final price?"
    )

    print(f"\nRunning the same harness with {len(BACKBONES)} different backbones on:\n  {task}\n")
    results = compare_backbones(task, BACKBONES)

    for result in results:
        print(f"--- backbone: {result['model']} ---")
        print(f"answer: {result['answer']}")
        print(
            f"tokens: {result['input_tokens']} in / {result['output_tokens']} out "
            f"| time: {result['seconds']}s\n"
        )


if __name__ == "__main__":
    main()
