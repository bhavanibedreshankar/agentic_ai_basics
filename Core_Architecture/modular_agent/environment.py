"""
CONCEPT: Environment — the boundary between the agent's decisions and the
outside world. It's the one place that actually calls an Action's function,
and the one place responsible for turning a crash into a safe, reportable
result instead of an exception that kills the whole run.

Splitting this out from ActionRegistry (actions.py) matters once actions get
more realistic: ../../Safety_and_Control/sandboxing/ and
../../Tools_and_Actions/ show actions that touch the filesystem, the network,
or a subprocess. All of that risk and error-handling belongs here, at the
execution boundary — not spread across every Action's own function, and not
in loop.py, which should only ever see a clean ActionResult back.
"""

from __future__ import annotations

from dataclasses import dataclass

from actions import Action


@dataclass
class ActionResult:
    output: str
    is_error: bool = False


class Environment:
    """Executes actions on the agent's behalf and reports what happened."""

    def execute_action(self, action: Action, tool_input: dict) -> ActionResult:
        try:
            output = action.function(tool_input)
            return ActionResult(output=str(output), is_error=False)
        except Exception as exc:  # noqa: BLE001 - any action failure must reach the agent, not crash the loop
            return ActionResult(output=f"Error: {exc}", is_error=True)
