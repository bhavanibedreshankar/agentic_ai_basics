"""
CONCEPT: Dynamic Tool Selection — an agent that draws from a large pool of
tool "libraries" but only loads the few tools relevant to the CURRENT
request into its context, instead of sending every tool's full definition
on every call.

This uses Anthropic's native `tool_search_tool_bm25_20251119` server
tool — declared for real here, not hand-rolled, the same "use the real
mechanism when one exists" approach as
../../Tools_and_Actions/web_search/web_search.py's `web_search`
declaration. No beta header is required; it's generally available on the
Claude API. The mechanism: every tool below is still sent in full in the
`tools` array on every single request (the API needs the complete schema
server-side to search over and to expand matches into) but marked
`defer_loading: True` — that flag controls what enters Claude's CONTEXT,
not what you transmit over the wire. Only the search tool itself starts
loaded; a deferred tool's definition never reaches the model's attention
until a search actually surfaces it.

Contrast with ../tool_registry/basic_tool_registry.py: that registry's
`build_tool_catalog()` sends its ENTIRE catalog into context on every
call — fine for 5 tools, but that file's own closing note admits this
doesn't scale, and names tool search as the fix. This template IS that
fix: the same registry idea (name, description, schema, and handler
declared together in one place) applied at a scale where sending
everything up front would waste tokens and hurt tool-selection accuracy
(Anthropic's own docs cite selection accuracy degrading past 30-50 tools
sent up front).

Also contrast with ../../Dynamic_Agent_Spawning/dynamic_agent_spawning/:
that template invents a new AGENT PERSONA at runtime. This template
selects among EXISTING, pre-written tools at runtime — nothing here is
synthesized; the ten tools below were all written in advance, only which
ones enter context is decided per-request.

The libraries: weather, finance, calendar, email, github, and notes — 10
tools total, deliberately spanning unrelated domains so no single query
needs more than 2-3 of them. Watch the `[tool search]` and `[discovered]`
lines in the output to see exactly which tools got loaded for each
request — it's never all 10.

Use case: a general assistant with a large personal-productivity
toolbox. Type 'exit' to quit.
"""

from __future__ import annotations

import ast
import operator
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a personal productivity assistant with tools across several "
    "areas: weather, finance, calendar, email, GitHub, and notes. Search "
    "for the tools you need before using them — don't assume a tool "
    "exists without checking."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: the search tool itself must stay non-deferred (defer_loading
# defaults to False) — the API rejects a request where every tool,
# including the search tool, is deferred. BM25 takes natural-language
# queries; the alternative tool_search_tool_regex_20251119 variant takes
# Python regex patterns instead — BM25 reads more naturally in a demo
# transcript, which is why it's used here.
TOOL_SEARCH_TOOL = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}

# CONCEPT: registry-style single source of truth (same idea as
# ../tool_registry/basic_tool_registry.py's TOOL_REGISTRY) — but every
# entry also carries defer_loading: True, so none of these schemas sit in
# Claude's context until a search actually surfaces them. A real
# deployment would leave its 3-5 most-used tools non-deferred for
# efficiency (per Anthropic's guidance); this demo defers all ten
# instead, so every discovery is visible in the printed output.
DEFERRED_TOOLS = [
    {
        "name": "weather_get_current",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Austin'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
        "defer_loading": True,
    },
    {
        "name": "finance_convert_currency",
        "description": "Convert an amount from one currency to another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "3-letter code, e.g. 'USD'"},
                "to_currency": {"type": "string", "description": "3-letter code, e.g. 'EUR'"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        "defer_loading": True,
    },
    {
        "name": "finance_get_stock_price",
        "description": "Get the current price of a stock by ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "e.g. 'AAPL'"}},
            "required": ["ticker"],
        },
        "defer_loading": True,
    },
    {
        "name": "calendar_schedule_event",
        "description": "Schedule a calendar event with a title and date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "ISO date, e.g. '2026-08-01'"},
            },
            "required": ["title", "date"],
        },
        "defer_loading": True,
    },
    {
        "name": "email_send",
        "description": "Send an email to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
        "defer_loading": True,
    },
    {
        "name": "github_lookup_issue",
        "description": "Look up a GitHub issue by repository and issue number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "e.g. 'anthropics/anthropic-sdk-python'"},
                "issue_number": {"type": "integer"},
            },
            "required": ["repo", "issue_number"],
        },
        "defer_loading": True,
    },
    {
        "name": "utils_calculate",
        "description": "Evaluate a basic arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '2 + 2 * 3'"}},
            "required": ["expression"],
        },
        "defer_loading": True,
    },
    {
        "name": "utils_convert_units",
        "description": "Convert a value between units of measurement (length, weight, temperature).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "description": "e.g. 'miles', 'kg', 'fahrenheit'"},
                "to_unit": {"type": "string", "description": "e.g. 'km', 'lb', 'celsius'"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        "defer_loading": True,
    },
    {
        "name": "notes_save",
        "description": "Save a personal note with a title and content.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
            "required": ["title", "content"],
        },
        "defer_loading": True,
    },
    {
        "name": "notes_search",
        "description": "Search previously saved personal notes by keyword.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "defer_loading": True,
    },
]

# CONCEPT: sent in full on EVERY request regardless of what gets
# discovered — the API needs every deferred tool's schema server-side to
# search over and to expand tool_reference blocks into later.
# defer_loading changes what enters Claude's CONTEXT, not what you
# transmit; see the module docstring.
TOOLS = [TOOL_SEARCH_TOOL, *DEFERRED_TOOLS]

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    """Restricted arithmetic evaluator — never pass model-influenced
    strings straight to eval(), same reasoning as
    ../tool_registry/basic_tool_registry.py's `_safe_eval`.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Mock implementations for every deferred tool. Deliberately simple and
# deterministic — the concept under test here is DISCOVERY, not any one
# tool's real-world integration (for that, see
# ../../Tools_and_Actions/api_connectors_mcp/).
# ---------------------------------------------------------------------------
_MOCK_WEATHER = {"austin": (91, "sunny"), "chicago": (68, "cloudy"), "seattle": (61, "rainy")}
_MOCK_RATES = {("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("EUR", "USD"): 1.09}
_MOCK_STOCKS = {"AAPL": 231.50, "GOOG": 178.20, "MSFT": 415.30}
_NOTES: dict[str, str] = {}


def weather_get_current(location: str, unit: str = "fahrenheit") -> str:
    temp_f, condition = _MOCK_WEATHER.get(location.lower(), (72, "clear"))
    temp = temp_f if unit == "fahrenheit" else round((temp_f - 32) * 5 / 9)
    return f"{location}: {temp}°{'F' if unit == 'fahrenheit' else 'C'}, {condition}"


def finance_convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    rate = _MOCK_RATES.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return f"No rate available for {from_currency} -> {to_currency}"
    return f"{amount} {from_currency.upper()} = {round(amount * rate, 2)} {to_currency.upper()}"


def finance_get_stock_price(ticker: str) -> str:
    price = _MOCK_STOCKS.get(ticker.upper())
    return f"{ticker.upper()}: ${price}" if price else f"No price data for {ticker.upper()}"


def calendar_schedule_event(title: str, date: str) -> str:
    return f"Scheduled '{title}' on {date}"


def email_send(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} (subject: '{subject}')"


def github_lookup_issue(repo: str, issue_number: int) -> str:
    return f"{repo}#{issue_number}: 'Example issue title' (open)"


def utils_calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval").body
    return f"{expression} = {_safe_eval(tree)}"


def utils_convert_units(value: float, from_unit: str, to_unit: str) -> str:
    if (from_unit, to_unit) == ("fahrenheit", "celsius"):
        return f"{value}°F = {round((value - 32) * 5 / 9, 2)}°C"
    if (from_unit, to_unit) == ("celsius", "fahrenheit"):
        return f"{value}°C = {round(value * 9 / 5 + 32, 2)}°F"
    conversions = {
        ("miles", "km"): 1.60934,
        ("km", "miles"): 0.62137,
        ("kg", "lb"): 2.20462,
        ("lb", "kg"): 0.45359,
    }
    factor = conversions.get((from_unit, to_unit))
    if factor is None:
        return f"No conversion available for {from_unit} -> {to_unit}"
    return f"{value} {from_unit} = {round(value * factor, 2)} {to_unit}"


def notes_save(title: str, content: str) -> str:
    _NOTES[title] = content
    return f"Saved note '{title}'"


def notes_search(query: str) -> str:
    matches = [t for t, c in _NOTES.items() if query.lower() in t.lower() or query.lower() in c.lower()]
    return f"Found notes: {matches}" if matches else "No matching notes"


_HANDLERS = {
    "weather_get_current": weather_get_current,
    "finance_convert_currency": finance_convert_currency,
    "finance_get_stock_price": finance_get_stock_price,
    "calendar_schedule_event": calendar_schedule_event,
    "email_send": email_send,
    "github_lookup_issue": github_lookup_issue,
    "utils_calculate": utils_calculate,
    "utils_convert_units": utils_convert_units,
    "notes_save": notes_save,
    "notes_search": notes_search,
}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Dispatch for DISCOVERED, CLIENT-SIDE tools only. The search tool
    itself has no handler here — it runs entirely on Anthropic's
    servers and never appears as a `tool_use` block (see run_turn).
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}", True
    try:
        return str(handler(**tool_input)), False
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to Claude
        return f"Error: {exc}", True


def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        # CONCEPT: append the full response.content UNCHANGED, including
        # the server_tool_use and tool_search_tool_result blocks — the
        # API needs those exact blocks back to keep expanding discovered
        # tools correctly in later turns without re-searching.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "server_tool_use":
                # Claude's call to the SEARCH tool itself — runs on
                # Anthropic's servers. Never send a tool_result for this
                # block's id.
                query = block.input.get("query") or block.input.get("pattern", "")
                print(f"\n  [tool search] {query!r}")
            elif block.type == "tool_search_tool_result":
                if block.content.type == "tool_search_tool_search_result":
                    names = [ref.tool_name for ref in block.content.tool_references]
                    print(f"  [discovered] {names or 'no matches'}")
                else:
                    print(f"  [search error] {block.content.error_code}: {block.content.error_message}")
            elif block.type == "text":
                print(f"\nAssistant: {block.text}")
            elif block.type == "tool_use":
                # A DISCOVERED, client-side tool — execute and return a
                # tool_result exactly like any other custom tool.
                print(f"  [calling] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [result] {result_text}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        if response.stop_reason != "tool_use":
            return

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Dynamic tool selection demo — 10 tools across 6 libraries, only the relevant ones load per request.")
    print("Type 'exit' to quit.\n")
    print("Try: \"What's the weather in Austin, and convert 100 miles to km.\"\n")

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
