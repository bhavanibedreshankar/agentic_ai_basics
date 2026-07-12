"""
CONCEPT: Sub-agent / Worker Agent — a specialized agent that executes one
well-defined subtask, packaged as a reusable, independently-callable
unit.

../orchestrator/'s delegate_to_X functions are each a single, non-looping
API call — enough when a subtask can be done in one shot. This template
zooms into what a worker looks like when it CAN'T: a `WorkerAgent` is a
complete agent in its own right, with its own system prompt, its own
tools, and its own internal tool-calling loop (structurally identical to
`../../Core_Architecture/tool_use/basic_agentic_tools.py`'s run_turn) —
it just happens to be wrapped as one `.run(task)` call another system can
invoke without knowing or caring how many turns it took internally.

That's the actual point of the worker/sub-agent pattern: composability.
An orchestrator (or a supervisor, or a handoff target) doesn't need to
know whether the thing it's calling is a single API call or a five-turn
agentic loop with its own tools — from the outside, both look identical:
call it with a task string, get a result string back.

Two concrete workers are defined below — a data_analyst (with a
calculator tool) and a researcher (with a mock fact-lookup tool) — each
independently runnable from the CLI. Type 'exit' to quit.
"""

from __future__ import annotations

import ast
import operator
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ---------------------------------------------------------------------------
# CONCEPT: the worker itself — a self-contained agent, not just a prompt.
# ---------------------------------------------------------------------------
class WorkerAgent:
    def __init__(self, name: str, system_prompt: str, tools: list[dict], tool_executor):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = tool_executor  # (name, input) -> (result_text, is_error)

    def run(self, task: str) -> str:
        """Run this worker on a task end to end, including as many
        internal tool-calling rounds as it needs, and return its final
        answer. Everything that happens inside this method — how many
        turns it takes, which tools it calls, in what order — is
        invisible to whatever called .run(); it's this worker's own
        business.
        """
        messages = [{"role": "user", "content": task}]
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self.tools,
                output_config={"effort": EFFORT},
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text" and response.stop_reason == "tool_use":
                    # Interim reasoning before a tool call — print it so
                    # the worker's internal process is visible in this
                    # demo, even though a caller using .run() wouldn't
                    # normally see it.
                    print(f"    [{self.name} thinking] {block.text}")

            if response.stop_reason != "tool_use":
                return "".join(block.text for block in response.content if block.type == "text")

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"    [{self.name} action] {block.name}({block.input})")
                    result_text, is_error = self.tool_executor(block.name, block.input)
                    print(f"    [{self.name} observation] {result_text}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                            "is_error": is_error,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Worker 1: data_analyst — has a calculator tool.
# ---------------------------------------------------------------------------
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval").body
    return f"{expression} = {_safe_eval(tree)}"


def data_analyst_tools(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "calculate":
        return calculate(**tool_input), False
    return f"Unknown tool: {name}", True


data_analyst = WorkerAgent(
    name="data_analyst",
    system_prompt=(
        "You are a data analyst. Use the calculate tool for any arithmetic "
        "instead of computing it yourself. Show your final answer clearly."
    ),
    tools=[
        {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 + 2 * 3'"}},
                "required": ["expression"],
            },
        },
    ],
    tool_executor=data_analyst_tools,
)


# ---------------------------------------------------------------------------
# Worker 2: researcher — has a mock fact-lookup tool.
# ---------------------------------------------------------------------------
FACTS = {
    "population of france": "France has a population of approximately 68 million.",
    "population of germany": "Germany has a population of approximately 84 million.",
    "gdp of france": "France's GDP is approximately $3.0 trillion.",
    "gdp of germany": "Germany's GDP is approximately $4.5 trillion.",
}


def lookup_fact(topic: str) -> str:
    key = topic.strip().lower()
    if key in FACTS:
        return FACTS[key]
    key_words = set(key.split())
    best_topic, best_overlap = None, 0
    for known_topic in FACTS:
        overlap = len(key_words & set(known_topic.split()))
        if overlap > best_overlap:
            best_topic, best_overlap = known_topic, overlap
    if best_topic:
        return FACTS[best_topic]
    return f"No fact found for '{topic}'. Known topics: {', '.join(FACTS)}"


def researcher_tools(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "lookup_fact":
        return lookup_fact(**tool_input), False
    return f"Unknown tool: {name}", True


researcher = WorkerAgent(
    name="researcher",
    system_prompt="You are a researcher. Use lookup_fact to find information rather than guessing.",
    tools=[
        {
            "name": "lookup_fact",
            "description": "Look up a known fact by topic, e.g. 'population of france'.",
            "input_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "The topic to look up"}},
                "required": ["topic"],
            },
        },
    ],
    tool_executor=researcher_tools,
)

WORKERS = {"data_analyst": data_analyst, "researcher": researcher}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Worker agent demo — available workers: {', '.join(WORKERS)}. Type 'exit' to quit.\n")
    print("Try: \"data_analyst: what's 15% of 340 plus 22\"")
    print("Try: \"researcher: what's the combined GDP of France and Germany\"\n")

    while True:
        user_input = input("worker_name: task > ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input or ":" not in user_input:
            print(f"Format: <worker_name>: <task>. Available workers: {', '.join(WORKERS)}")
            continue

        worker_name, _, task = user_input.partition(":")
        worker_name = worker_name.strip()
        task = task.strip()

        if worker_name not in WORKERS:
            print(f"Unknown worker '{worker_name}'. Available: {', '.join(WORKERS)}")
            continue

        result = WORKERS[worker_name].run(task)
        print(f"\n{worker_name} final answer: {result}\n")


if __name__ == "__main__":
    main()
