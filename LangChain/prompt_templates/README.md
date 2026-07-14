# prompt_templates

Reusable, parameterized prompts that separate instructions from user input.

## prompt_templates.py

A support-ticket triage assistant. The same `ChatPromptTemplate` is reused across every ticket — only `customer_name`, `product`, and `ticket_text` change per call. Type `exit` to end the session.

### Concepts covered

- **`ChatPromptTemplate.from_messages`** — builds a reusable, role-tagged prompt from `(role, template_string)` pairs, with `{placeholders}` parsed out automatically as `.input_variables`.
- **`.partial()`** — bakes a constant (`support_tier`) into a template at definition time, so callers only supply the genuinely per-call fields (see `PARTIAL_TRIAGE_PROMPT`).
- **`PromptTemplate.from_template`** — the plain-text (non-chat) counterpart, for pipeline steps that don't need conversational roles.
- **`build_triage_chain`** — composes the template directly with a chat model (`TRIAGE_PROMPT | llm`), the seed of the LCEL composition explored fully in [`../chains/README.md`](../chains/README.md).
- Contrast with [`../../Core_Architecture/basics/basic.py`](../../Core_Architecture/basics/README.md) and [`../../Safety_and_Control/audit_trail/audit_trail.py`](../../Safety_and_Control/audit_trail/README.md), which both build prompts as hand-written f-strings with no validation that every slot got filled.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/prompt_templates/prompt_templates.py
```

Try:

```
Product (or 'exit'): mobile app
Ticket text: App crashes every time I open the camera tab

Triage:
Summary: App crashes on camera tab open.
Priority: high
Route to: technical
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `TRIAGE_PROMPT` / `SUPPORT_TIER_PROMPT` — edit the template text or add new `{placeholders}` directly; `ChatPromptTemplate` picks them up automatically

### See also

- [`../chains/README.md`](../chains/README.md) — composes a template into a full multi-step LCEL pipeline
- [`../output_parsers/README.md`](../output_parsers/README.md) — turns a template's raw text output into a validated object
