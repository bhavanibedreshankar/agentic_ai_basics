"""
CONCEPT: Dynamic Agent Spawning — a meta-agent capable of creating a
brand-new sub-agent AT RUNTIME (deciding its role, persona, and system
prompt on the fly) and assigning it a task, rather than delegating among
a fixed roster of specialists someone wrote into the code in advance.

Contrast with ../../Multi_Agent_Systems/orchestrator/orchestrator.py: its
three specialists — researcher, writer, editor — and their system
prompts are hardcoded constants in that file. The orchestrator can only
ever pick among those three tools, no matter what a request actually
needs. Here there is exactly ONE tool, spawn_subagent, and the role,
persona, and system prompt it produces come from the PARENT model's
tool_input at call time — the set of possible specialists is unbounded
and decided per-request, not by whoever wrote this file.

Contrast also with ../../Multi_Agent_Systems/worker_agent/worker_agent.py:
its WorkerAgent instances (data_analyst, researcher) are pre-instantiated,
with fixed tools, before main() ever runs. A sub-agent here doesn't exist
until the parent decides it needs one and constructs it inside the same
call that uses it — it has no persistent identity and is discarded the
moment its task returns; asking for "the same role" again later just
builds an equivalent instance from scratch, not a resumed one.

Unbounded dynamic creation is a real cost/safety risk — a parent that
keeps deciding "I need one more specialist" could fan out indefinitely,
each spawn costing its own API call. MAX_SUBAGENTS_PER_TURN caps how many
a single parent turn may create, enforced in code (dispatch()), never
left to the model to self-limit — the same "code decides the hard rule"
idea as PASS_THRESHOLD in
../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py
and MAX_RULES in
../../Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py.

Use case: a meta-agent with NO built-in specialists, only the ability to
invent them — give it a request spanning several kinds of expertise it
couldn't have been pre-configured for (a home-bakery lease touching
contract law, health code, and small-business tax) and watch it spawn
several unique, never-hardcoded sub-agents to cover it. Type 'exit' to
quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SUBAGENT_MAX_TOKENS = 1024

# CONCEPT: a hard cap on how many sub-agents ONE parent turn may spawn —
# see the module docstring for why this has to be a code-enforced limit.
MAX_SUBAGENTS_PER_TURN = 4

META_AGENT_SYSTEM_PROMPT = (
    "You are a meta-agent with no built-in specialists of your own. When "
    "a request needs expertise you don't have, invent a specialist for "
    "it: call spawn_subagent with a specific role (e.g. 'Health Code "
    "Inspector', not 'Assistant'), a one-sentence persona describing what "
    "that role should focus on and how it should answer, and the exact "
    "task to hand it. Spawn as many DIFFERENT specialists as the request "
    "genuinely needs — don't reuse one generic role for unrelated "
    "questions — then combine their answers into your final response."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SPAWN_SUBAGENT_TOOL = {
    "name": "spawn_subagent",
    "description": (
        "Create a brand-new, single-use sub-agent with a specific role "
        "and persona, and assign it one task. The sub-agent exists only "
        "for this call and has no memory of anything outside the task "
        "you give it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "description": "A short job title for the sub-agent, e.g. 'Trademark Attorney'"},
            "persona": {"type": "string", "description": "One or two sentences describing that role's expertise and how it should answer"},
            "task": {"type": "string", "description": "The specific task or question to hand this sub-agent"},
        },
        "required": ["role", "persona", "task"],
    },
}


def spawn_subagent(role: str, persona: str, task: str) -> str:
    """CONCEPT: the sub-agent's entire identity — its system prompt — is
    ASSEMBLED FROM STRINGS THE PARENT MODEL CHOSE, not selected from a
    fixed set written into this file. This one function can become a tax
    accountant, a health inspector, or a marine biologist, depending
    entirely on the tool_input the parent supplies at runtime.
    """
    system_prompt = f"You are a {role}. {persona} Answer the task directly and concisely."
    response = client.messages.create(
        model=MODEL,
        max_tokens=SUBAGENT_MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": task}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name != "spawn_subagent":
        return f"Unknown tool: {name}", True
    return spawn_subagent(**tool_input), False


def dispatch(name: str, tool_input: dict, spawn_count: int) -> tuple[str, bool, int]:
    """Enforces MAX_SUBAGENTS_PER_TURN before letting a spawn through.
    Kept separate from execute_tool so the cap can be unit-tested without
    an API call, the same split used by
    ../../Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py's
    evolve()/propose_rule().
    """
    if spawn_count >= MAX_SUBAGENTS_PER_TURN:
        return (
            f"MAX_SUBAGENTS_PER_TURN={MAX_SUBAGENTS_PER_TURN} reached for this turn — "
            "answer using the specialists you already spawned instead of creating another.",
            True,
            spawn_count,
        )
    result_text, is_error = execute_tool(name, tool_input)
    return result_text, is_error, spawn_count + 1


def run_turn(messages: list[dict]) -> None:
    spawn_count = 0
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=META_AGENT_SYSTEM_PROMPT,
            tools=[SPAWN_SUBAGENT_TOOL],
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nMeta-agent: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                role = block.input.get("role", "?")
                print(f"  [spawn request: {role}] task: {block.input.get('task', '')}")
                result_text, is_error, spawn_count = dispatch(block.name, block.input, spawn_count)
                print(f"  [{role} -> ] {result_text[:150]}...")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Meta-agent with no built-in specialists — it invents them at runtime. Type 'exit' to quit.\n")
    print(
        "Try: \"I'm signing a lease to run a bakery out of a rented commercial "
        "kitchen — what should I watch out for?\"\n"
    )

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
