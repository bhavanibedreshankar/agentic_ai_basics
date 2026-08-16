"""
CONCEPT: Fixture and Mock Management — extracting the boilerplate for
building canned LLM responses and disposable tool-side-effect state into
small, NAMED, reusable builders, instead of hand-rolling
`types.SimpleNamespace(...)` and a fresh dict literal inline in every single
test function.

Look at how the previous two templates in this directory got their fake
data:
  - ../tool_unit_testing/tool_unit_testing.py's `fresh_orders()` is a
    one-off fixture function, defined and used only in that one file.
  - ../agent_integration_testing/agent_integration_testing.py's
    `text_response`, `tool_use_response`, and `FakeClient` are a second,
    slightly different hand-rolled set, defined and used only in THAT file.

Both work fine at 5-10 tests in one file. They stop working once a suite
grows across many test files (exactly the scaling problem this whole topic
exists to answer) — every new file either re-implements the same
`types.SimpleNamespace(type="text", ...)` boilerplate slightly differently,
or copy-pastes it, and the two copies quietly drift out of sync with the
real `anthropic` response shape over time (a field gets renamed upstream,
one copy gets updated, the other doesn't, and a test silently stops proving
what it claims to prove).

This template is that shared toolbox, built once and reused: `text_block`,
`tool_use_block`, and `response` are the atomic builders every fake Claude
response is assembled from; `FakeClient` is the reusable queue-based stub
(same idea as ../agent_integration_testing/'s `FakeClient`, but now the ONE
copy the rest of a growing suite would import); `fresh_orders_db` is the
same dependency-injected fixture idea as
../tool_unit_testing/tool_unit_testing.py's `fresh_orders`, generalized to
take overrides. The tests below don't just exercise these builders in
isolation — `test_composing_fixtures_for_integration_scenario` shows the
payoff: a full multi-step orchestration scenario assembled in about five
lines, instead of the twenty-plus lines it took inline in
../agent_integration_testing/agent_integration_testing.py.

Pytest-style test module: `pytest fixture_and_mock_management.py` or
`python3 fixture_and_mock_management.py` (see `_run_tests_manually`).
"""

from __future__ import annotations

import inspect
import sys
import types

# ---------------------------------------------------------------------------
# CONCEPT: response builders — thin wrappers around types.SimpleNamespace
# that match the shape of a real anthropic.types.Message (`.content`,
# `.stop_reason`, and per-block `.type`/`.text` or `.type`/`.name`/`.input`/
# `.id`). Centralizing these three lines means if the real SDK's shape ever
# changes, there is exactly ONE place to update, not one place per test
# file.
# ---------------------------------------------------------------------------
def text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, tool_input: dict, block_id: str = "tu_1") -> types.SimpleNamespace:
    return types.SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(blocks: list, stop_reason: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)


def text_response(text: str) -> types.SimpleNamespace:
    """The common case: a single text block, conversation-ending response."""
    return response([text_block(text)], stop_reason="end_turn")


def tool_use_response(*calls: tuple[str, dict, str]) -> types.SimpleNamespace:
    """`calls` is (name, input, id) triples — pass more than one to build a
    response requesting several tools in parallel.
    """
    blocks = [tool_use_block(name, inp, block_id) for name, inp, block_id in calls]
    return response(blocks, stop_reason="tool_use")


class FakeClient:
    """A queue-based stand-in for anthropic.Anthropic(). Pre-load the exact
    sequence of responses (or exceptions, to simulate a transient failure —
    see ../agent_integration_testing/) a test wants the model to produce.
    `self.calls` records every kwargs dict passed to `.create(...)`, so a
    test can assert on what was actually SENT (which tools were declared,
    what the message history looked like at call time), not just what came
    back.
    """

    class _Messages:
        def __init__(self, outer: "FakeClient") -> None:
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls.append(kwargs)
            if not self._outer._responses:
                raise RuntimeError("FakeClient ran out of queued responses")
            next_item = self._outer._responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = FakeClient._Messages(self)


# ---------------------------------------------------------------------------
# CONCEPT: state fixtures — same dependency-injection idea as
# ../tool_unit_testing/tool_unit_testing.py's fresh_orders(), generalized to
# accept overrides so a test can start from a specific scenario (e.g. "an
# order that's already outside the refund window") without hand-writing the
# whole dict every time.
# ---------------------------------------------------------------------------
def fresh_orders_db(overrides: dict | None = None) -> dict:
    base = {
        "1001": {"item": "Wireless Mouse", "price": 25.00, "days_since_purchase": 10},
        "1002": {"item": "Bluetooth Speaker", "price": 60.00, "days_since_purchase": 45},
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# CONCEPT: the tests below fall into two groups — first, proving the
# fixtures themselves behave correctly (a broken fixture silently breaks
# every test built on it, so fixtures need their own tests too); second,
# showing the composed payoff of having them.
# ---------------------------------------------------------------------------
def test_text_response_shape() -> None:
    r = text_response("hello")
    assert r.stop_reason == "end_turn"
    assert r.content[0].type == "text"
    assert r.content[0].text == "hello"


def test_tool_use_response_supports_multiple_calls() -> None:
    r = tool_use_response(("lookup_order", {"order_id": "1001"}, "a"), ("lookup_order", {"order_id": "1002"}, "b"))
    assert r.stop_reason == "tool_use"
    assert len(r.content) == 2
    assert {b.id for b in r.content} == {"a", "b"}


def test_fake_client_returns_queued_responses_in_order() -> None:
    client = FakeClient([text_response("first"), text_response("second")])
    first = client.messages.create(messages=[])
    second = client.messages.create(messages=[])
    assert first.content[0].text == "first"
    assert second.content[0].text == "second"


def test_fake_client_raises_queued_exceptions() -> None:
    client = FakeClient([RuntimeError("boom")])
    try:
        client.messages.create(messages=[])
        assert False, "expected the queued RuntimeError to be raised"
    except RuntimeError as exc:
        assert "boom" in str(exc)


def test_fake_client_records_call_arguments() -> None:
    client = FakeClient([text_response("ok")])
    client.messages.create(messages=[{"role": "user", "content": "hi"}], tools=["fake_tool"])
    assert len(client.calls) == 1
    assert client.calls[0]["tools"] == ["fake_tool"]
    assert client.calls[0]["messages"][0]["content"] == "hi"


def test_fresh_orders_db_is_isolated_between_calls() -> None:
    a = fresh_orders_db()
    b = fresh_orders_db()
    a["1001"]["price"] = 999.00
    assert b["1001"]["price"] == 25.00


def test_fresh_orders_db_supports_overrides() -> None:
    db = fresh_orders_db(overrides={"1003": {"item": "Webcam", "price": 45.00, "days_since_purchase": 5}})
    assert db["1003"]["item"] == "Webcam"
    assert "1001" in db, "overrides should extend the base fixture, not replace it"


def test_composing_fixtures_for_integration_scenario() -> None:
    # CONCEPT: the payoff. This mirrors
    # ../agent_integration_testing/agent_integration_testing.py's
    # `test_single_tool_call_then_final_answer` almost exactly, but built
    # from the shared toolbox above instead of that file's own hand-rolled
    # copies — a new test file gets this scenario in five lines, not twenty.
    client = FakeClient(
        [tool_use_response(("lookup_order", {"order_id": "1001"}, "tu_1")), text_response("Found your Wireless Mouse order.")]
    )
    orders = fresh_orders_db()

    first = client.messages.create(messages=[{"role": "user", "content": "look up order 1001"}])
    assert first.stop_reason == "tool_use"
    order_id = first.content[0].input["order_id"]
    assert orders[order_id]["item"] == "Wireless Mouse"

    second = client.messages.create(messages=[])
    assert second.stop_reason == "end_turn"
    assert len(client.calls) == 2


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
    print("Fixture/mock toolbox tests — plus a demo of composing them into a bigger scenario cheaply.\n")
    _run_tests_manually()
