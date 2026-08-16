"""
CONCEPT: Tool Unit Testing — testing individual tool FUNCTIONS in isolation,
with no LLM call involved at all, covering both the happy path and the
error/edge cases a real caller (an LLM deciding what arguments to pass) can
and will hit.

Every tool-using template elsewhere in this repo (e.g.
../../Core_Architecture/tool_use/basic_agentic_tools.py,
../../Benchmarking/trace_evaluation/trace_evaluation.py) writes its tool
functions against MODULE-LEVEL global state — `basic_agentic_tools.py`
reads/writes a `TASKS_FILE` on disk, `trace_evaluation.py` reads a module-
level `ORDERS` dict. That's the right shape for a running demo (state should
persist across turns), but it's the WRONG shape for a unit test: two tests
that both mutate the same global dict or file can interfere with each other,
and tests can't easily set up a specific starting state without reaching
into module internals.

This template reimplements the same order-lookup/refund domain as
`trace_evaluation.py`, but with one deliberate change: every tool function
takes its data store as an explicit parameter (`orders: dict`) instead of
reading a module global. This is DEPENDENCY INJECTION, and it's what makes
these functions unit-testable in the first place — each test builds its own
fresh, disposable `orders` dict via `fresh_orders()`, so tests can run in any
order, in parallel, or a thousand times over, without ever stepping on each
other's state.

This is deliberately NOT structured as a chat demo like most templates here
— it's a pytest-style test module: plain `test_*` functions using bare
`assert`, runnable either with `pytest tool_unit_testing.py` or directly with
`python3 tool_unit_testing.py` (see `_run_tests_manually` at the bottom,
which needs no pytest installed at all).

See also: this is the FAST, cheapest-per-run layer of the test pyramid built
in ../test_suite_pyramid/test_suite_pyramid.py, and contrasts with
../../Benchmarking/ (which SCORES an agent's live output quality) — this
file never calls the Claude API at all, it tests plain Python.
"""

from __future__ import annotations

import inspect
import sys

# ---------------------------------------------------------------------------
# The tools under test — same domain as ../../Benchmarking/trace_evaluation/
# trace_evaluation.py's ORDERS/lookup_order/check_refund_eligibility/
# issue_refund, but rewritten to take their store as an argument instead of
# reading a module global. REFUND_WINDOW_DAYS stays a module constant since
# it's a fixed business rule, not per-test state.
# ---------------------------------------------------------------------------
REFUND_WINDOW_DAYS = 30


def fresh_orders() -> dict:
    """CONCEPT: a fixture function — called fresh at the top of every test
    so each test gets its own independent dict. Nothing here is shared
    across tests, which is what makes test order and parallelism safe.
    This same idea, generalized into a reusable module of its own, becomes
    ../fixture_and_mock_management/fixture_and_mock_management.py.
    """
    return {
        "1001": {"item": "Wireless Mouse", "price": 25.00, "days_since_purchase": 10},
        "1002": {"item": "Bluetooth Speaker", "price": 60.00, "days_since_purchase": 45},
    }


def lookup_order(orders: dict, order_id: str) -> dict:
    if order_id not in orders:
        raise ValueError(f"No order found with id '{order_id}'")
    return {"order_id": order_id, **orders[order_id]}


def check_refund_eligibility(orders: dict, order_id: str) -> dict:
    order = lookup_order(orders, order_id)
    eligible = order["days_since_purchase"] <= REFUND_WINDOW_DAYS
    return {"order_id": order_id, "eligible": eligible, "days_since_purchase": order["days_since_purchase"]}


def issue_refund(orders: dict, order_id: str, amount: float) -> dict:
    # CONCEPT: input validation belongs IN the tool, not only in the model's
    # prompt instructions — the model can and will occasionally pass a bad
    # amount (a negative number, a typo'd extra zero), and a tool that
    # trusts its input blindly is a bug waiting for a unit test to catch it.
    # See ../adversarial_safety_testing/adversarial_safety_testing.py for
    # the same idea pushed further, against deliberately hostile input.
    order = lookup_order(orders, order_id)
    if amount <= 0:
        raise ValueError(f"Refund amount must be positive, got {amount}")
    if amount > order["price"]:
        raise ValueError(f"Refund amount {amount} exceeds order price {order['price']}")
    return {"order_id": order_id, "refunded_amount": amount, "status": "refunded"}


# ---------------------------------------------------------------------------
# CONCEPT: unit tests — one behavior per test, named so a failure tells you
# exactly what broke without reading the assertion. Both the happy path AND
# the error cases are tested; a tool that only ever gets tested on valid
# input is only half-tested, since a live agent will eventually call it with
# something malformed or out of range.
# ---------------------------------------------------------------------------
def test_lookup_order_found() -> None:
    orders = fresh_orders()
    result = lookup_order(orders, "1001")
    assert result["item"] == "Wireless Mouse"
    assert result["order_id"] == "1001"


def test_lookup_order_not_found_raises() -> None:
    orders = fresh_orders()
    try:
        lookup_order(orders, "9999")
        assert False, "expected ValueError for unknown order id"
    except ValueError as exc:
        assert "9999" in str(exc)


def test_check_refund_eligibility_within_window() -> None:
    orders = fresh_orders()
    result = check_refund_eligibility(orders, "1001")  # 10 days old
    assert result["eligible"] is True


def test_check_refund_eligibility_outside_window() -> None:
    orders = fresh_orders()
    result = check_refund_eligibility(orders, "1002")  # 45 days old
    assert result["eligible"] is False


def test_check_refund_eligibility_boundary_day() -> None:
    # CONCEPT: boundary testing — exactly REFUND_WINDOW_DAYS should still
    # count as eligible ("<=", not "<"). Off-by-one errors on boundaries
    # like this are exactly the kind of bug a happy-path-only test suite
    # never catches, because the happy path never lands exactly on the edge.
    orders = fresh_orders()
    orders["1003"] = {"item": "Keyboard", "price": 40.00, "days_since_purchase": REFUND_WINDOW_DAYS}
    result = check_refund_eligibility(orders, "1003")
    assert result["eligible"] is True


def test_issue_refund_success() -> None:
    orders = fresh_orders()
    result = issue_refund(orders, "1001", 25.00)
    assert result["status"] == "refunded"
    assert result["refunded_amount"] == 25.00


def test_issue_refund_rejects_negative_amount() -> None:
    orders = fresh_orders()
    try:
        issue_refund(orders, "1001", -5.00)
        assert False, "expected ValueError for negative amount"
    except ValueError as exc:
        assert "positive" in str(exc)


def test_issue_refund_rejects_zero_amount() -> None:
    orders = fresh_orders()
    try:
        issue_refund(orders, "1001", 0)
        assert False, "expected ValueError for zero amount"
    except ValueError:
        pass


def test_issue_refund_rejects_amount_over_price() -> None:
    orders = fresh_orders()
    try:
        issue_refund(orders, "1001", 25.01)  # order 1001 costs exactly 25.00
        assert False, "expected ValueError for amount exceeding order price"
    except ValueError as exc:
        assert "exceeds" in str(exc)


def test_fresh_orders_is_isolated_between_calls() -> None:
    # CONCEPT: proving the fixture itself is safe to reuse across many
    # tests — mutating one call's dict must never leak into another's.
    a = fresh_orders()
    b = fresh_orders()
    a["1001"]["price"] = 999.00
    assert b["1001"]["price"] == 25.00, "fresh_orders() must return independent dicts, not a shared reference"


# ---------------------------------------------------------------------------
# CONCEPT: a dependency-free test runner. This module is fully pytest-
# discoverable as-is (`pytest tool_unit_testing.py`) — the block below just
# means it also runs standalone with plain `python3`, so this repo doesn't
# force pytest as a hard requirement to try the concept out.
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
    print("Tool unit tests — no LLM call, no ANTHROPIC_API_KEY needed.\n")
    _run_tests_manually()
