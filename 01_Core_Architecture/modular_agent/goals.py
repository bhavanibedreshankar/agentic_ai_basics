"""
CONCEPT: Goal — the thing an agent is working toward, held as data instead of
being baked into a hand-written system prompt string.

../agent/agent.py hardcodes its purpose directly into SYSTEM_PROMPT: the goal
and the agent's instructions are one fused string. That's fine for a single
demo, but it means "what the agent is trying to achieve" can't be inspected,
swapped, or composed without editing prose. Here a Goal is a small, plain
piece of data; something else (language.py's construct_system_prompt) is
responsible for turning a list of Goals into an actual prompt. Add a second
goal, reorder goals, or generate them at runtime, and nothing downstream
(loop.py, agent.py) has to change — they only ever see `agent.goals`, a list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Goal:
    """One thing the agent is trying to achieve.

    name: short label, shown to the model as a heading.
    description: the actual instruction/objective text for this goal.
    """

    name: str
    description: str
