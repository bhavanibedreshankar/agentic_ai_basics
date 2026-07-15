"""
CONCEPT: the agent loop itself — perceive (read the model's response),
reason (the model decides what it means), plan (the model picks a next
action or decides it's done), act (Environment executes that action) —
repeated until the agent is satisfied or MAX_ITERATIONS is hit. Same cycle
as ../agent/agent.py's pursue_goal(), but rewritten to depend only on the
public surface of the other files in this package (agent.goals,
agent.memory, agent.actions, agent.environment, agent.language, agent.client)
instead of anything concrete.

This is the file this whole package's modularity is FOR: run_agent_loop()
never mentions "calculate", "terminate", a specific memory format, or a
specific prompt string by name. Add a new Action in builtin_actions.py, add
a second Goal, swap Memory for a persistent implementation, or point
language.py at a different provider — none of it touches a single line
here. If you ever find yourself editing this file to add a *feature*
(as opposed to fixing the loop mechanics themselves), that's a sign the
feature belongs in one of the other files instead.
"""

from __future__ import annotations

from environment import ActionResult


def run_agent_loop(agent, user_input: str, max_iterations: int = 8) -> str:
    """Run `agent` to completion on one input, with no further human input,
    and return its final answer."""

    agent.memory.add(role="user", content=user_input, kind="user_input")

    for step in range(1, max_iterations + 1):
        system = agent.language.construct_system_prompt(agent.goals)
        tools = agent.language.construct_tools(agent.actions)
        messages = agent.language.construct_messages(agent.memory)

        response = agent.client.messages.create(
            model=agent.model,
            max_tokens=agent.max_tokens,
            system=system,
            tools=tools,
            output_config={"effort": agent.effort},
            messages=messages,
        )
        agent.memory.add(role="assistant", content=response.content, kind="assistant_turn")

        # The agent decides for itself when it's done — nothing here tells
        # it the goal is met. Same check as ../agent/agent.py.
        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            print(f"[step {step}] agent finished on its own\n")
            return final_text

        tool_results = []
        final_answer = None
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[step {step}] reasoning: {block.text.strip()}")
            if block.type != "tool_use":
                continue

            print(f"[step {step}] act: {block.name}({block.input})")
            action = agent.actions.get_action(block.name)
            if action is None:
                result = ActionResult(output=f"Unknown action: {block.name}", is_error=True)
            else:
                result = agent.environment.execute_action(action, block.input)
            print(f"[step {step}] perceive: {result.output}\n")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.output,
                    "is_error": result.is_error,
                }
            )

            # A terminal action (see builtin_actions.py's `terminate`) ends
            # the loop as soon as it succeeds, without waiting for the model
            # to stop requesting tools on its own.
            if action is not None and action.terminal and not result.is_error:
                final_answer = result.output

        agent.memory.add(role="user", content=tool_results, kind="action_result")
        if final_answer is not None:
            return final_answer

    # Unconditional backstop — same idea as ../../Execution_Loops/max_iterations/.
    # Autonomy without a hard cap is dangerous: a confused agent could loop
    # on tool calls indefinitely, burning tokens and money.
    return f"(stopped after {max_iterations} steps without reaching a final answer)"
