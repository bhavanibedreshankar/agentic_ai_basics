"""
CONCEPT: Action / ActionRegistry — everything the agent is capable of doing,
declared as data (name, description, JSON Schema, a function) and looked up
by name, rather than an if/elif chain the loop has to know about.

Contrast with ../agent/agent.py's execute_tool(), which is a hardcoded
`if name == "calculate": ... elif name == "record_finding": ...` dispatcher —
adding a new tool there means editing that function, and by extension the
file the loop lives in. Same idea as ../../Agent_Frameworks_and_Patterns/
tool_registry/basic_tool_registry.py, applied here specifically so the
*agent loop* (loop.py) never has to change when the *action catalog* grows.

To add a new capability to the agent: write a function with signature
`(tool_input: dict) -> str`, wrap it in an Action, and register() it —
see builtin_actions.py for real examples. loop.py only ever calls
`registry.get_action(name)`; it has no idea `calculate` or `terminate` exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Action:
    """One capability the agent can invoke via a tool call.

    name/description/parameters: exactly what the LLM API's `tools` field
        needs (parameters is a JSON Schema dict, i.e. an `input_schema`).
    function: the actual Python callable, `(tool_input: dict) -> str`.
        Raise on failure; Environment (environment.py) is what catches it.
    terminal: if True, a successful call to this action ends the agent loop
        and its output becomes the agent's final answer — see
        builtin_actions.py's `terminate` for the canonical example.
    """

    name: str
    description: str
    parameters: dict
    function: Callable[[dict], str]
    terminal: bool = False


@dataclass
class ActionRegistry:
    """A name -> Action lookup table. The full catalog of things the agent
    can currently do, built up by whoever assembles the agent (see main.py),
    never by the loop itself.
    """

    _actions: Dict[str, Action] = field(default_factory=dict)

    def register(self, action: Action) -> None:
        self._actions[action.name] = action

    def get_action(self, name: str) -> Action | None:
        return self._actions.get(name)

    def all_actions(self) -> List[Action]:
        return list(self._actions.values())
