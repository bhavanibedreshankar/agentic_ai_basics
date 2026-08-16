"""
CONCEPT: Trace Evaluation — scoring an agent's entire EXECUTION TRACE (every
tool call, tool result, and intermediate step it took) instead of only its
final answer.

Every scoring template elsewhere in this directory looks at ONE thing: the
final text an agent produced.
  - `../task_accuracy_eval/task_accuracy_eval.py` string/number-matches the
    final answer against a known-correct value.
  - `../llm_judge_benchmarking/llm_judge_benchmarking.py`'s `evaluate_output`
    sends (task, FINAL output, rubric) to a judge model.

Both of those are blind to HOW the agent got there. A tool-using agent can
produce a perfectly worded final answer while its trace is a mess underneath
— it called `issue_refund` before ever checking eligibility, it looked up the
same order three times, or its final sentence states a price that never
actually appeared in any tool result. Conversely, a trace can look completely
correct step-by-step and still end in a final answer that ignores what the
tools returned. Final-output scoring alone can't tell these apart; trace
evaluation looks at the sequence, not just the last line.

This template captures a real trace from a live, tool-using support agent
(same tool-calling loop shape as ../../01_Core_Architecture/tool_use/basic_agentic_tools.py),
then scores that trace with TWO complementary mechanisms:

  1. RULE-BASED checks (`check_trace_rules`) — deterministic, no model call,
     instant and free. Catches structural problems a judge might miss or
     might not catch reliably: redundant tool calls, too many steps
     (efficiency), and a domain-specific precondition violation (refunding
     an order without ever checking it was eligible). This is the same
     "pass/fail decided in code" philosophy as `evaluator_agent.py`'s
     `PASS_THRESHOLD`, just applied to trace STRUCTURE instead of a score.
  2. LLM-AS-JUDGE over the whole trace (`judge_trace`) — same structured-
     output judge shape as `llm_judge_benchmarking.py`'s `evaluate_output`,
     but the prompt is the full formatted step sequence, not just the final
     answer, so it can score groundedness (did the final answer only state
     facts that actually came from a tool result?) and task adherence
     (did the agent solve what was actually asked?) — judgments that
     require seeing the intermediate steps, not just the ending.

This also contrasts with ../../17_LangChain/callbacks_and_tracing/callbacks_and_tracing.py
and ../../10_Safety_and_Control/audit_trail/audit_trail.py: those two are about
*capturing* a trace as a chain/agent runs. This template assumes a trace has
already been captured and asks the separate question "was this trace any
good?"

Use case: a support agent with order-lookup/eligibility/refund tools, run
once live for real, plus one hand-built BAD_TRACE (an agent that skips the
eligibility check) to show both checks catching a real defect on demand
rather than only when the live model happens to misbehave. Type 'exit' to
quit.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import anthropic

# --- API settings (see ../../01_Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 512
EFFORT = "medium"
PASS_THRESHOLD = 7  # out of 10, same convention as evaluator_agent.py / llm_judge_benchmarking.py

# Rule-based efficiency ceiling: more tool calls than this on a simple
# support task is flagged as inefficient, independent of whether the
# final answer happens to be correct.
MAX_TOOL_CALLS = 4

SYSTEM_PROMPT = (
    "You are a customer support agent. Use the available tools to look up "
    "orders, check refund eligibility, and issue refunds. ALWAYS check "
    "refund eligibility before issuing a refund. Explain your reasoning "
    "briefly in your final answer."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# A tiny mock order database — stand-in for a real orders/refunds API.
# ---------------------------------------------------------------------------
ORDERS = {
    "1001": {"item": "Wireless Mouse", "price": 25.00, "days_since_purchase": 10},
    "1002": {"item": "Bluetooth Speaker", "price": 60.00, "days_since_purchase": 45},
}
REFUND_WINDOW_DAYS = 30

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order's item and price by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_refund_eligibility",
        "description": "Check whether an order is still within the refund window. Call this BEFORE issue_refund.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order. Only call this after confirming eligibility.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    },
]


def lookup_order(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise ValueError(f"No order found with id '{order_id}'")
    return {"order_id": order_id, **ORDERS[order_id]}


def check_refund_eligibility(order_id: str) -> dict:
    order = lookup_order(order_id)
    eligible = order["days_since_purchase"] <= REFUND_WINDOW_DAYS
    return {"order_id": order_id, "eligible": eligible, "days_since_purchase": order["days_since_purchase"]}


def issue_refund(order_id: str, amount: float) -> dict:
    # NOTE: deliberately no eligibility enforcement here, on purpose — the
    # tool itself trusts its caller, exactly like the untrusted-input tools
    # elsewhere in the repo. Whether the AGENT checked eligibility first is
    # a question about the agent's behavior, not the tool's, which is
    # precisely what `check_trace_rules` below exists to catch.
    return {"order_id": order_id, "refunded_amount": amount, "status": "refunded"}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name == "lookup_order":
            return json.dumps(lookup_order(**tool_input)), False
        if name == "check_refund_eligibility":
            return json.dumps(check_refund_eligibility(**tool_input)), False
        if name == "issue_refund":
            return json.dumps(issue_refund(**tool_input)), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to Claude
        return f"Error: {exc}", True


# ---------------------------------------------------------------------------
# CONCEPT: the trace itself — a plain, ordered record of what happened,
# independent of both the Claude SDK's message format and of whatever
# scores it later. Any agent framework's log could be normalized into this
# same shape.
# ---------------------------------------------------------------------------
@dataclass
class TraceStep:
    kind: str  # "tool_call" | "tool_result" | "final_response"
    tool_name: str | None = None
    tool_input: dict | None = None
    text: str | None = None


def format_trace(trace: list[TraceStep]) -> str:
    """Render the trace as plain text for the judge model to read — the
    trace-evaluation equivalent of llm_judge_benchmarking.py handing its
    judge a single `output` string, except here it's the whole sequence.
    """
    lines = []
    for i, step in enumerate(trace, start=1):
        if step.kind == "tool_call":
            lines.append(f"{i}. CALL {step.tool_name}({step.tool_input})")
        elif step.kind == "tool_result":
            lines.append(f"{i}. RESULT from {step.tool_name}: {step.text}")
        elif step.kind == "final_response":
            lines.append(f"{i}. FINAL ANSWER: {step.text}")
    return "\n".join(lines)


def run_agent(task: str) -> list[TraceStep]:
    """Run the real tool-calling loop and capture every step as a TraceStep.

    Same inner-loop shape as ../../01_Core_Architecture/tool_use/basic_agentic_tools.py's
    run_turn — the only difference is that each tool call/result/final answer
    is also appended to `trace`, so we have a complete record to hand to the
    evaluators afterward instead of just printing as we go.
    """
    trace: list[TraceStep] = []
    messages: list[dict] = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            trace.append(TraceStep(kind="final_response", text=final_text))
            return trace

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                trace.append(TraceStep(kind="tool_call", tool_name=block.name, tool_input=block.input))
                result_text, is_error = execute_tool(block.name, block.input)
                trace.append(TraceStep(kind="tool_result", tool_name=block.name, text=result_text))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text, "is_error": is_error}
                )

        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# CONCEPT: rule-based trace checks — deterministic, no LLM call. These run
# in microseconds and never disagree with themselves on a re-run, which an
# LLM judge (see judge_trace below) cannot promise.
# ---------------------------------------------------------------------------
@dataclass
class RuleViolation:
    rule: str
    detail: str


def check_trace_rules(trace: list[TraceStep]) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    calls = [s for s in trace if s.kind == "tool_call"]

    # Rule 1: efficiency — too many tool calls for a simple task suggests
    # looping or indecision, regardless of whether the final answer is fine.
    if len(calls) > MAX_TOOL_CALLS:
        violations.append(
            RuleViolation("excessive_steps", f"{len(calls)} tool calls, exceeds MAX_TOOL_CALLS={MAX_TOOL_CALLS}")
        )

    # Rule 2: redundant calls — the exact same tool called with the exact
    # same arguments more than once wastes steps and cost for no new
    # information.
    seen = set()
    for c in calls:
        key = (c.tool_name, json.dumps(c.tool_input, sort_keys=True))
        if key in seen:
            violations.append(RuleViolation("redundant_call", f"{c.tool_name}({c.tool_input}) called more than once"))
        seen.add(key)

    # Rule 3: domain precondition — issue_refund must be preceded by a
    # check_refund_eligibility call for the SAME order_id. This is the kind
    # of structural, sequence-dependent bug an LLM judge reading only the
    # final answer (like llm_judge_benchmarking.py's evaluate_output) would
    # never see, since the final answer can read as perfectly reasonable.
    checked_eligible_orders = {
        c.tool_input.get("order_id") for c in calls if c.tool_name == "check_refund_eligibility"
    }
    for c in calls:
        if c.tool_name == "issue_refund" and c.tool_input.get("order_id") not in checked_eligible_orders:
            violations.append(
                RuleViolation(
                    "precondition_skipped",
                    f"issue_refund called for order {c.tool_input.get('order_id')} with no prior check_refund_eligibility call",
                )
            )

    return violations


# ---------------------------------------------------------------------------
# CONCEPT: LLM-as-judge over the FULL trace — structurally identical to
# llm_judge_benchmarking.py's evaluate_output (structured score + code-side
# pass/fail against PASS_THRESHOLD), but the thing being judged is the whole
# formatted step sequence, which lets it score qualities no single-output
# judge can see.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of AI agent execution traces. You "
    "will see a task and the agent's full trace of tool calls, tool results, "
    "and final answer. Score three things from 0 to 10: "
    "groundedness (does the final answer state ONLY facts that actually "
    "appear in a tool result, with no invented numbers or claims?), "
    "task_adherence (did the agent's actions and final answer actually "
    "address what the task asked?), and error_recovery (if any tool result "
    "indicated a problem, ineligibility, or error, did the agent handle it "
    "sensibly instead of ignoring it?). Explain your scores in one or two "
    "sentences of specific feedback."
)

TRACE_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness": {"type": "integer", "description": "0 to 10"},
        "task_adherence": {"type": "integer", "description": "0 to 10"},
        "error_recovery": {"type": "integer", "description": "0 to 10"},
        "feedback": {"type": "string"},
    },
    "required": ["groundedness", "task_adherence", "error_recovery", "feedback"],
    "additionalProperties": False,
}


@dataclass
class TraceJudgment:
    groundedness: int
    task_adherence: int
    error_recovery: int
    average: float
    passed: bool
    feedback: str


def judge_trace(task: str, trace: list[TraceStep]) -> TraceJudgment:
    prompt = f"Task: {task}\n\nAgent trace:\n{format_trace(trace)}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        system=JUDGE_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": TRACE_JUDGMENT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    average = (result["groundedness"] + result["task_adherence"] + result["error_recovery"]) / 3
    # CONCEPT: same as evaluator_agent.py — pass/fail is a CODE decision
    # against a fixed threshold, never left to the model itself.
    return TraceJudgment(
        groundedness=result["groundedness"],
        task_adherence=result["task_adherence"],
        error_recovery=result["error_recovery"],
        average=average,
        passed=average >= PASS_THRESHOLD,
        feedback=result["feedback"],
    )


@dataclass
class TraceEvaluation:
    rule_violations: list[RuleViolation] = field(default_factory=list)
    judgment: TraceJudgment | None = None


def evaluate_trace(task: str, trace: list[TraceStep]) -> TraceEvaluation:
    """Run both evaluators and combine them. The two are independent and
    complementary: rules catch structural defects instantly and for free;
    the judge catches semantic defects (groundedness, adherence) that no
    amount of rule-writing can fully enumerate in advance.
    """
    return TraceEvaluation(rule_violations=check_trace_rules(trace), judgment=judge_trace(task, trace))


def print_evaluation(trace: list[TraceStep], evaluation: TraceEvaluation) -> None:
    print("\n--- trace ---")
    print(format_trace(trace))

    print("\n--- rule-based checks ---")
    if evaluation.rule_violations:
        for v in evaluation.rule_violations:
            print(f"  [VIOLATION] {v.rule}: {v.detail}")
    else:
        print("  no violations")

    j = evaluation.judgment
    print("\n--- LLM-as-judge over the trace ---")
    print(f"  groundedness={j.groundedness}/10  task_adherence={j.task_adherence}/10  error_recovery={j.error_recovery}/10")
    print(f"  average={j.average:.1f}/10  passed={j.passed}")
    print(f"  feedback: {j.feedback}")


# ---------------------------------------------------------------------------
# A hand-built BAD trace: an agent that refunds order 1002 (outside the
# 30-day window) without ever calling check_refund_eligibility, then claims
# in its final answer that the order "was confirmed eligible" — a fact that
# never appeared in any tool result. This demonstrates both evaluators
# catching a real defect on demand, rather than only when the live model
# happens to misbehave (which it usually won't, given SYSTEM_PROMPT).
# ---------------------------------------------------------------------------
BAD_TASK = "Refund order 1002, the customer says it arrived damaged."
BAD_TRACE = [
    TraceStep(kind="tool_call", tool_name="lookup_order", tool_input={"order_id": "1002"}),
    TraceStep(kind="tool_result", tool_name="lookup_order", text=json.dumps(lookup_order("1002"))),
    TraceStep(kind="tool_call", tool_name="issue_refund", tool_input={"order_id": "1002", "amount": 60.00}),
    TraceStep(kind="tool_result", tool_name="issue_refund", text=json.dumps(issue_refund("1002", 60.00))),
    TraceStep(
        kind="final_response",
        text="Order 1002 was confirmed eligible for a refund and $60.00 has been refunded.",
    ),
]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Trace evaluation demo — scoring an agent's full trace, not just its final answer.\n")

    print("=== Demo 1: a real, live agent run ===")
    task = "I'd like a refund for order 1001, it stopped working."
    trace = run_agent(task)
    evaluation = evaluate_trace(task, trace)
    print_evaluation(trace, evaluation)

    print("\n=== Demo 2: a hand-built BAD trace (skipped eligibility check, ungrounded claim) ===")
    bad_evaluation = evaluate_trace(BAD_TASK, BAD_TRACE)
    print_evaluation(BAD_TRACE, bad_evaluation)

    print("\n=== Now try your own task (type 'exit' to quit) ===")
    while True:
        task = input("\nTask: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue
        trace = run_agent(task)
        evaluation = evaluate_trace(task, trace)
        print_evaluation(trace, evaluation)


if __name__ == "__main__":
    main()
