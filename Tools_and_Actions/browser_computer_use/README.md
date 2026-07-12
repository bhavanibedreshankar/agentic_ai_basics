# browser_computer_use

Browser / Computer Use — agents that can control a browser or desktop UI autonomously: read what's on screen, decide an action, act, observe the result, repeat.

## browser_computer_use.py

A browser automation agent working through a mock checkout flow (browse a product, add to cart, fill in shipping details, submit) using structured page state instead of pixel screenshots. Type `exit` to quit.

### Concepts covered

- **Structured state instead of pixels, and why** — Anthropic's native computer use tool (`computer_20251124`) works from real screenshot images and coordinate-based clicks, which needs a real running desktop or browser behind the API to capture from — infrastructure this template doesn't have. This template demonstrates the other genuinely-used real pattern instead: the agent reads a text description of what elements exist on the current mock page (an accessibility-tree/DOM-like view) and acts on them by ID, not coordinate. It's a different real approach, not a watered-down version of computer use.
- **`MockBrowser`** — a small state machine: pages are fixed, but cart contents and form data are dynamic, so `read_page()` always reflects what's actually happened in the conversation so far.
- **The read → decide → act → observe loop** — `SYSTEM_PROMPT` explicitly tells the model to call `read_page` after every navigation, since (like a real browser) nothing in this template assumes the model can predict what a page contains without checking.
- **Guard conditions surfaced as tool errors, not crashes** — checking out with an empty cart, or submitting the order form before both fields are filled, return clear error strings the model can react to, rather than raising exceptions that would kill the loop.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Tools_and_Actions/browser_computer_use/browser_computer_use.py
```

Try:

```
You: Buy a coffee mug and check out with name Jane Doe and email jane@example.com.

  [action] read_page({})
  [observation] Page: home
Elements:
  [product_mug] (link) Coffee Mug - $12.99
  ...
  [action] click({'element_id': 'product_mug'})
  ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `MockBrowser._static_elements` — the page definitions; add pages or elements to extend the flow

### See also

- `../../tool_use/README.md` — the client-side tool-calling loop this template's `read_page`/`click`/`type_text` tools follow
- `../file_io_tools/README.md` — another client-executed tool set, for contrast with this one's dynamic mock-state backend
