"""
CONCEPT: this file IS the extension point. Everything above it (goals.py,
memory.py, actions.py, environment.py, language.py, loop.py, agent.py) is
fixed architecture; this file is where new capabilities actually get added.

To give the agent a new ability: write a function shaped `(tool_input: dict)
-> str` that raises on failure, wrap it in an Action with a name/description/
JSON Schema, and register() it in register_builtin_actions(). Nothing else
in this package needs to know it exists — loop.py finds it purely by name
via ActionRegistry.get_action().
"""

from __future__ import annotations

import ast
import operator

from actions import Action, ActionRegistry

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    # Never eval() a model-influenced string directly — restrict to basic
    # arithmetic on a parsed expression tree. Same approach as
    # ../agent/agent.py and ../../Agent_Frameworks_and_Patterns/tool_registry/.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(tool_input: dict) -> str:
    expression = tool_input["expression"]
    tree = ast.parse(expression, mode="eval").body
    return f"{expression} = {_safe_eval(tree)}"


def record_finding(tool_input: dict) -> str:
    # No real computation — its only purpose is to make the agent's planning
    # visible as tool calls in the transcript, same as ../agent/agent.py.
    return f"Noted: {tool_input['finding']}"


def terminate(tool_input: dict) -> str:
    # Marked terminal=True below, so a successful call to this action ends
    # the loop immediately with this text as the final answer — an explicit
    # alternative to waiting for stop_reason != "tool_use".
    return tool_input["final_answer"]


def register_builtin_actions(registry: ActionRegistry) -> None:
    registry.register(
        Action(
            name="calculate",
            description="Evaluate a basic arithmetic expression. Call this for any math instead of computing it yourself.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g. '12.50 + 7.25 + 19.99'"},
                },
                "required": ["expression"],
            },
            function=calculate,
        )
    )
    registry.register(
        Action(
            name="record_finding",
            description=(
                "Record an intermediate finding or conclusion you've reached so far, before continuing. "
                "Call this after each meaningful step of your plan, not just at the end."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "finding": {"type": "string", "description": "A short statement of what you've just figured out"},
                },
                "required": ["finding"],
            },
            function=record_finding,
        )
    )
    registry.register(
        Action(
            name="terminate",
            description="Call this once you have a final, complete answer to the goal, passing that answer.",
            parameters={
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string", "description": "The complete final answer to the goal"},
                },
                "required": ["final_answer"],
            },
            function=terminate,
            terminal=True,
        )
    )
