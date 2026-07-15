"""
CONCEPT: AgentLanguage — the single seam between the agent's internal state
(Goals, Memory, ActionRegistry) and the wire format a specific LLM API
expects (system string, tools array, messages array).

Every other file in this package is deliberately provider-agnostic: Goal is
just a name/description, Memory just stores role/content/kind, Action is
just a name/description/schema/function. Nothing about "Anthropic's Messages
API shape" leaks into any of them — it's all concentrated here, behind one
abstract interface. Point to a different provider (or a second one, for
../../Model_Routing/-style fallback) by writing a new AgentLanguage subclass
and handing it to Agent; loop.py, agent.py, memory.py, actions.py,
environment.py, and goals.py never need to change.

Honest simplification: Memory (memory.py) already stores each item's content
pre-shaped for Anthropic's API (plain text for user turns, content-block
lists for assistant turns and tool results), so construct_messages() below
is closer to a passthrough than a real translation. A from-scratch,
truly provider-agnostic version would keep Memory storing plain structured
facts ("agent said X", "tool Y returned Z") and do the *entire* translation
into blocks here. This template keeps that translation inline in loop.py
for readability — the point being demonstrated is the seam's location, not
building out multiple providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from actions import ActionRegistry
from goals import Goal
from memory import Memory


class AgentLanguage(ABC):
    @abstractmethod
    def construct_system_prompt(self, goals: List[Goal]) -> str:
        """Turn the agent's goals into the fixed instruction text for this run."""

    @abstractmethod
    def construct_tools(self, actions: ActionRegistry) -> List[dict]:
        """Turn the current action catalog into this API's `tools` schema."""

    @abstractmethod
    def construct_messages(self, memory: Memory) -> List[dict]:
        """Turn stored memory into this API's `messages` list."""


class AnthropicToolCallingLanguage(AgentLanguage):
    """The concrete translation for Claude's Messages API tool-calling format."""

    def construct_system_prompt(self, goals: List[Goal]) -> str:
        lines = [
            "You are an autonomous agent working toward the goal(s) below, not a "
            "chat assistant answering one message. You will get no further input "
            "from a human until you finish. Break each goal into steps yourself, "
            "use the available tools to work through them, and only stop calling "
            "tools once you're ready to give a final, complete answer.",
            "",
            "Goals:",
        ]
        lines.extend(f"- {goal.name}: {goal.description}" for goal in goals)
        return "\n".join(lines)

    def construct_tools(self, actions: ActionRegistry) -> List[dict]:
        return [
            {
                "name": action.name,
                "description": action.description,
                "input_schema": action.parameters,
            }
            for action in actions.all_actions()
        ]

    def construct_messages(self, memory: Memory) -> List[dict]:
        return memory.as_message_list()
