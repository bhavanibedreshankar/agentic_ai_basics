"""
CONCEPT: Browser / Computer Use — an agent that controls a browser or
desktop UI autonomously: reading what's on screen, deciding an action
(click, type, navigate), and observing the result, repeatedly, until a
task is done.

Anthropic's native computer use tool (`computer_20251124` at the time of
writing) does this via PIXELS: Claude receives actual screenshot images
and issues coordinate-based actions (click at x,y; type; scroll) — the
same way a human would use a completely unfamiliar app, with no special
access to its internals. That needs a real, running desktop or browser
behind the API (a VM, a headless browser, a virtual display) to capture
screenshots from and execute clicks against — infrastructure this
template doesn't have and won't fake.

What this template demonstrates instead is the OTHER real pattern for
browser automation: STRUCTURED STATE instead of pixels. Rather than a
screenshot, the agent reads a description of what's on the mock page —
what elements exist, their type, their label — and acts on elements by
ID rather than by coordinate. This isn't a simplification of computer
use; it's a genuinely different, commonly used approach in real browser
agents (an accessibility tree or a page's DOM serves the same role a
screenshot does for pixel-based agents), and it's the one that's actually
runnable here: every tool below is a plain Python function operating on
an in-memory mock browser, client-side, same shape as
`../../Core_Architecture/tool_use/basic_agentic_tools.py`.

Demonstrated on a mock checkout flow: browse a product, add it to cart,
and check out with shipping details — a multi-step task requiring
several read -> decide -> act -> observe cycles. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a browser automation agent. You interact with a web app "
    "through a page tool: read_page to see what's currently on screen, "
    "click to activate a button or link by its id, and type_text to fill "
    "a text field by its id. Always call read_page after navigating or "
    "clicking to see the new state before deciding your next action — "
    "never assume what a page contains without reading it first."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ---------------------------------------------------------------------------
# CONCEPT: the mock browser. A small state machine standing in for a real
# browser's DOM — pages are defined statically, but cart/form state is
# dynamic, so read_page reflects what's actually happened so far, not a
# fixed script.
# ---------------------------------------------------------------------------
class MockBrowser:
    def __init__(self) -> None:
        self.current_page = "home"
        self.cart: list[str] = []
        self.form_data: dict[str, str] = {}
        self.order_placed = False

    def _static_elements(self, page: str) -> list[dict]:
        pages = {
            "home": [
                {"id": "product_mug", "type": "link", "label": "Coffee Mug - $12.99"},
                {"id": "product_shirt", "type": "link", "label": "T-Shirt - $19.99"},
                {"id": "go_cart", "type": "link", "label": f"View Cart ({len(self.cart)} items)"},
            ],
            "product_mug": [
                {"id": "add_to_cart", "type": "button", "label": "Add Coffee Mug to Cart"},
                {"id": "go_home", "type": "link", "label": "Back to Home"},
            ],
            "product_shirt": [
                {"id": "add_to_cart", "type": "button", "label": "Add T-Shirt to Cart"},
                {"id": "go_home", "type": "link", "label": "Back to Home"},
            ],
            "cart": [
                {"id": "go_checkout", "type": "button", "label": "Proceed to Checkout"},
                {"id": "go_home", "type": "link", "label": "Continue Shopping"},
            ],
            "checkout": [
                {"id": "name_field", "type": "text_input", "label": f"Full Name (current: '{self.form_data.get('name_field', '')}')"},
                {"id": "email_field", "type": "text_input", "label": f"Email (current: '{self.form_data.get('email_field', '')}')"},
                {"id": "submit_order", "type": "button", "label": "Submit Order"},
            ],
            "confirmation": [
                {"id": "go_home", "type": "link", "label": "Back to Home"},
            ],
        }
        return pages.get(page, [])

    def read_page(self) -> str:
        elements = self._static_elements(self.current_page)
        lines = [f"Page: {self.current_page}"]
        if self.current_page == "cart":
            lines.append(f"Cart contents: {self.cart or 'empty'}")
        if self.current_page == "confirmation":
            lines.append(f"Order placed: {self.order_placed}, shipped to: {self.form_data}")
        lines.append("Elements:")
        for el in elements:
            lines.append(f"  [{el['id']}] ({el['type']}) {el['label']}")
        return "\n".join(lines)

    def click(self, element_id: str) -> str:
        valid_ids = {el["id"] for el in self._static_elements(self.current_page)}
        if element_id not in valid_ids:
            return f"Error: no element '{element_id}' on the current page ({self.current_page})"

        if element_id == "product_mug":
            self.current_page = "product_mug"
        elif element_id == "product_shirt":
            self.current_page = "product_shirt"
        elif element_id == "add_to_cart":
            item = "Coffee Mug" if self.current_page == "product_mug" else "T-Shirt"
            self.cart.append(item)
            return f"Added {item} to cart. Cart now has {len(self.cart)} item(s)."
        elif element_id == "go_cart":
            self.current_page = "cart"
        elif element_id == "go_home":
            self.current_page = "home"
        elif element_id == "go_checkout":
            if not self.cart:
                return "Error: cart is empty, cannot check out"
            self.current_page = "checkout"
        elif element_id == "submit_order":
            if "name_field" not in self.form_data or "email_field" not in self.form_data:
                return "Error: name and email must both be filled in before submitting"
            self.order_placed = True
            self.current_page = "confirmation"

        return f"Clicked '{element_id}'. Now on page: {self.current_page}"

    def type_text(self, element_id: str, text: str) -> str:
        valid_ids = {el["id"] for el in self._static_elements(self.current_page) if el["type"] == "text_input"}
        if element_id not in valid_ids:
            return f"Error: no text field '{element_id}' on the current page ({self.current_page})"
        self.form_data[element_id] = text
        return f"Typed '{text}' into '{element_id}'"


TOOLS = [
    {
        "name": "read_page",
        "description": "Read the current page's content — what page you're on and what elements are available to interact with. Call this after every navigation or click.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "click",
        "description": "Click a button or link on the current page by its element id.",
        "input_schema": {
            "type": "object",
            "properties": {"element_id": {"type": "string", "description": "The id of the element to click, from read_page's output"}},
            "required": ["element_id"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into a text field on the current page by its element id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "The id of the text field, from read_page's output"},
                "text": {"type": "string", "description": "The text to type into the field"},
            },
            "required": ["element_id", "text"],
        },
    },
]


def execute_tool(name: str, tool_input: dict, browser: MockBrowser) -> tuple[str, bool]:
    try:
        if name == "read_page":
            return browser.read_page(), False
        if name == "click":
            return browser.click(**tool_input), False
        if name == "type_text":
            return browser.type_text(**tool_input), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}", True


def run_turn(messages: list[dict], browser: MockBrowser) -> None:
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

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [action] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input, browser)
                print(f"  [observation] {result_text}")
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
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Browser automation demo (mock checkout flow). Type 'exit' to quit.\n")
    print(
        "Try: \"Buy a coffee mug and check out with name Jane Doe and "
        "email jane@example.com.\"\n"
    )

    browser = MockBrowser()
    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages, browser)


if __name__ == "__main__":
    main()
