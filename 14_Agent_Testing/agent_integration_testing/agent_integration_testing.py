"""
CONCEPT: Agent Integration Testing — testing the FULL multi-turn, multi-tool
agentic loop end-to-end, with the LLM client swapped for a deterministic
stub, so the ORCHESTRATION logic itself (does the loop stop at the right
time, does message history grow correctly, do parallel tool calls all get
executed and matched back up, does a transient failure get retried) is
verified without ever making a real, billed, non-deterministic API call.

../tool_unit_testing/tool_unit_testing.py tests tool FUNCTIONS in isolation
— given these arguments, does `issue_refund` return the right thing. This
template tests something tool unit tests structurally cannot reach: the LOOP
that decides which tools to call, in what order, and when to stop. That loop
lives in `run_agent` below, built with the same shape as
../../01_Core_Architecture/tool_use/basic_agentic_tools.py's `run_turn` — the
only real difference is that `client` is a constructor PARAMETER here
instead of a module-level `anthropic.Anthropic()` singleton. That one change
(dependency injection again, same idea as ../tool_unit_testing/) is what
makes the loop testable: a test can hand `run_agent` a fake client that
returns whatever canned sequence of responses the test wants, with zero
network calls and zero API cost.

Contrast with ../../13_Benchmarking/trace_evaluation/trace_evaluation.py: that
template runs the SAME kind of loop for real and then scores the quality of
the resulting trace (was the answer grounded, was a precondition skipped).
This template runs the loop against FAKE, scripted responses specifically
chosen to exercise orchestration edge cases (a mid-run error, two tools
requested at once) that would be slow, expensive, and unreliable to
provoke from a real model on demand.

Pytest-style test module: `pytest agent_integration_testing.py` or
`python3 agent_integration_testing.py` (see `_run_tests_manually`).
"""

from __future__ import annotations

import inspect
import json
import sys
import types

# ---------------------------------------------------------------------------
# A small tool set — same domain as ../tool_unit_testing/tool_unit_testing.py
# and ../../13_Benchmarking/trace_evaluation/trace_evaluation.py, kept minimal
# here since the point of THIS file is the loop around the tools, not the
# tools themselves (those already have their own dedicated unit tests).
# ---------------------------------------------------------------------------
def lookup_order(order_id: str) -> dict:
    orders = {"1001": {"item": "Wireless Mouse", "price": 25.00}}
    if order_id not in orders:
        raise ValueError(f"No order found with id '{order_id}'")
    return {"order_id": order_id, **orders[order_id]}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name == "lookup_order":
            return json.dumps(lookup_order(**tool_input)), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return f"Error: {exc}", True


MAX_RETRIES = 2


def run_agent(client, task: str) -> list[dict]:
    """The loop under test. Same tool-calling shape as
    ../../01_Core_Architecture/tool_use/basic_agentic_tools.py's run_turn, plus
    one addition tested below: a transient API error retries the SAME call
    up to MAX_RETRIES times before giving up, instead of crashing the whole
    turn on the first hiccup.
    """
    messages: list[dict] = [{"role": "user", "content": task}]

    while True:
        response = _create_with_retry(client, messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return messages

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text, is_error = execute_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text, "is_error": is_error}
                )
        messages.append({"role": "user", "content": tool_results})


def _create_with_retry(client, messages: list[dict]):
    # CONCEPT: retrying a transient failure at the ORCHESTRATION layer, not
    # inside a tool. A dropped connection or a 500 from the API is not the
    # same kind of failure as a tool raising ValueError on bad input — it
    # means "we don't know what happened, try again," not "this specific
    # call was invalid." Retrying belongs here, once, rather than duplicated
    # inside every tool implementation.
    last_exc: Exception | None = None
    for _attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(messages=messages)
        except RuntimeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# CONCEPT: fixture builders for fake responses. Small and inline here on
# purpose — ../fixture_and_mock_management/fixture_and_mock_management.py
# is where this exact pattern gets extracted into a reusable, importable
# module once a suite has enough test files that copy-pasting it everywhere
# starts to hurt.
# ---------------------------------------------------------------------------
def text_response(text: str) -> types.SimpleNamespace:
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(content=[block], stop_reason="end_turn")


def tool_use_response(*calls: tuple[str, dict, str]) -> types.SimpleNamespace:
    """One response requesting one or more tool calls at once — `calls` is
    (tool_name, tool_input, tool_use_id) triples, so a test can build a
    PARALLEL multi-tool-call response by passing more than one.
    """
    blocks = [
        types.SimpleNamespace(type="tool_use", name=name, input=inp, id=call_id) for name, inp, call_id in calls
    ]
    return types.SimpleNamespace(content=blocks, stop_reason="tool_use")


class FakeClient:
    """CONCEPT: a queue-based stand-in for anthropic.Anthropic(). Each test
    pre-loads exactly the sequence of responses it wants `run_agent` to see,
    so the test controls the model's behavior completely instead of hoping
    a real model happens to do the thing being tested. `self.call_count`
    lets tests assert on HOW MANY times the API was actually called — the
    thing that proves retry logic ran, or that a loop didn't call the API
    more times than expected.
    """

    class _Messages:
        def __init__(self, outer: "FakeClient") -> None:
            self._outer = outer

        def create(self, **kwargs):
            self._outer.call_count += 1
            if not self._outer._responses:
                raise RuntimeError("FakeClient ran out of queued responses")
            next_item = self._outer._responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.messages = FakeClient._Messages(self)


# ---------------------------------------------------------------------------
# CONCEPT: orchestration tests. None of these care what a tool RETURNS
# (that's ../tool_unit_testing/'s job) — they care about the SHAPE of the
# conversation the loop produces: how many API calls happened, what ended up
# in `messages`, and whether the loop stopped at the right point.
# ---------------------------------------------------------------------------
def test_single_turn_no_tools_needed() -> None:
    client = FakeClient([text_response("Hi there!")])
    messages = run_agent(client, "hello")
    assert client.call_count == 1
    assert messages[-1]["role"] == "assistant"


def test_single_tool_call_then_final_answer() -> None:
    client = FakeClient(
        [
            tool_use_response(("lookup_order", {"order_id": "1001"}, "tu_1")),
            text_response("Your Wireless Mouse order was found."),
        ]
    )
    messages = run_agent(client, "look up order 1001")
    assert client.call_count == 2
    # CONCEPT: message history shape — a tool-use turn produces exactly
    # three new messages: the assistant's tool request, a user-role message
    # carrying the tool result, and the assistant's final answer.
    assert len(messages) == 4  # original user message + the 3 above
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["tool_use_id"] == "tu_1"
    assert "Wireless Mouse" in messages[2]["content"][0]["content"]
    assert messages[3]["role"] == "assistant"


def test_parallel_tool_calls_all_execute_and_match_ids() -> None:
    client = FakeClient(
        [
            tool_use_response(
                ("lookup_order", {"order_id": "1001"}, "tu_a"),
                ("lookup_order", {"order_id": "9999"}, "tu_b"),  # deliberately invalid, to prove errors don't crash the loop
            ),
            text_response("Here's what I found."),
        ]
    )
    messages = run_agent(client, "look up orders 1001 and 9999")
    tool_result_message = messages[2]["content"]
    assert len(tool_result_message) == 2, "both parallel tool calls must produce a result"
    ids = {r["tool_use_id"] for r in tool_result_message}
    assert ids == {"tu_a", "tu_b"}, "tool_use_id must round-trip so results match their calls"
    by_id = {r["tool_use_id"]: r for r in tool_result_message}
    assert by_id["tu_a"]["is_error"] is False
    assert by_id["tu_b"]["is_error"] is True  # unknown order id — surfaced as an error, not a crash


def test_transient_error_is_retried_and_succeeds() -> None:
    client = FakeClient([RuntimeError("connection reset"), text_response("recovered")])
    messages = run_agent(client, "hello")
    assert client.call_count == 2, "one failed attempt plus one successful retry"
    assert messages[-1]["role"] == "assistant"


def test_retries_exhausted_raises() -> None:
    client = FakeClient([RuntimeError("down"), RuntimeError("still down")])
    try:
        run_agent(client, "hello")
        assert False, "expected RuntimeError after exhausting MAX_RETRIES"
    except RuntimeError as exc:
        assert "after" in str(exc)
    assert client.call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# Dependency-free runner — see ../tool_unit_testing/tool_unit_testing.py for
# the same pattern explained.
# ---------------------------------------------------------------------------
def _run_tests_manually() -> None:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and inspect.isfunction(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    print("Agent integration tests — real orchestration loop, stubbed LLM client, no API calls.\n")
    _run_tests_manually()
