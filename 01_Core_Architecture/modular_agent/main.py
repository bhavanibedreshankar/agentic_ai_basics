"""
Entry point: wires up one concrete Agent from the modular pieces and runs
it on a goal, end to end. This is the file you'd copy and edit to build a
*different* agent — everything it imports stays fixed.

Same budget-checking use case as ../agent/agent.py, so you can compare the
two directly: that file is one script where prompt, tools, dispatch, and
loop are all inline; this one is the identical behavior assembled from
Goal + Memory + ActionRegistry + Environment + AgentLanguage + Agent.
"""

from __future__ import annotations

import os
import sys

import anthropic

from actions import ActionRegistry
from agent import Agent
from builtin_actions import register_builtin_actions
from environment import Environment
from goals import Goal
from language import AnthropicToolCallingLanguage


def build_agent() -> Agent:
    """Assemble one Agent instance from the modular pieces.

    This function is the whole "blueprint" in action: swap any one argument
    below (a different Goal, a different ActionRegistry, a persistent
    Memory, a different AgentLanguage) to get a differently-behaved agent
    without touching agent.py or loop.py at all.
    """
    registry = ActionRegistry()
    register_builtin_actions(registry)

    goals = [
        Goal(
            name="budget_check",
            description=(
                "Given a list of item prices and a budget, work out whether the items "
                "fit under budget, using the 'calculate' tool for any math and "
                "'record_finding' to note intermediate results. Call 'terminate' with "
                "your final answer once you're done."
            ),
        )
    ]

    return Agent(
        goals=goals,
        actions=registry,
        environment=Environment(),
        language=AnthropicToolCallingLanguage(),
        client=anthropic.Anthropic(),  # reads ANTHROPIC_API_KEY from the environment
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Modular agent blueprint. Give it one goal; it runs unattended until done.\n")
    goal_input = input("Goal: ").strip() or (
        "I want to buy 3 items priced at $12.50, $7.25, and $19.99. "
        "My budget is $50. Do they fit, and how much would I have left over?"
    )

    print(f"\nGoal: {goal_input}\n")
    agent = build_agent()
    answer = agent.run(goal_input)
    print(f"Final answer: {answer}")


if __name__ == "__main__":
    main()
