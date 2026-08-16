"""
CONCEPT: Agent — the assembly point. It doesn't implement goals, memory,
actions, environment execution, prompt construction, or the loop itself —
it just holds one of each and exposes `run()`. This is the class you
instantiate differently for every new use case (see main.py), while every
*other* file in this package stays untouched.

This split — assembly (Agent) vs. algorithm (loop.py) — is what makes the
package extensible without risk: building a new agent means writing a new
main.py-style script that constructs a different Agent(...), never editing
agent.py or loop.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import anthropic

from actions import ActionRegistry
from environment import Environment
from goals import Goal
from language import AgentLanguage
from loop import run_agent_loop
from memory import Memory


@dataclass
class Agent:
    goals: List[Goal]
    actions: ActionRegistry
    environment: Environment
    language: AgentLanguage
    client: anthropic.Anthropic
    memory: Memory = field(default_factory=Memory)

    # See ../basics/README.md for what each of these means.
    model: str = "claude-sonnet-5"
    max_tokens: int = 4096
    effort: str = "medium"

    def run(self, user_input: str, max_iterations: int = 8) -> str:
        """Pursue `user_input` to completion, unattended, and return the
        agent's final answer."""
        return run_agent_loop(self, user_input, max_iterations)
