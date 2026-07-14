# output_parsers

Coercing raw LLM text into structured, validated data (e.g. Pydantic models).

## output_parsers.py

Extracts a structured `TicketExtraction` (summary, priority, category) from free-form support ticket text, once via each of two methods, so their reliability difference is visible in the output rather than just claimed in a comment. Type `exit` to end the conversation.

### Concepts covered

- **`PydanticOutputParser`** — asks the model, via text instructions appended to the prompt (`get_format_instructions()`), to produce JSON matching a schema, then parses that text into a Pydantic model. Soft enforcement: malformed model output makes parsing raise.
- **`llm.with_structured_output(Model)`** — binds the schema as a tool the model must call, so arguments arrive pre-validated the way [`../../Core_Architecture/tool_use/basic_agentic_tools.py`](../../Core_Architecture/tool_use/README.md)'s `tool_input` does; no format instructions needed in the prompt.
- **`TicketExtraction`** — one Pydantic model, targeted by both methods, making the comparison apples-to-apples.
- The two `build_*_chain` functions show the parser/structured-output step slotting into a pipeline exactly like `StrOutputParser` does in [`../chains/chains.py`](../chains/README.md) — "turn the AIMessage into something else" is the same shape either way.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/output_parsers/output_parsers.py
```

Try:

```
Ticket text: App crashes every time I open the camera tab, please fix urgently

[PydanticOutputParser] TicketExtraction(summary='Camera tab crashes the app', priority='urgent', category='technical')
[with_structured_output] TicketExtraction(summary='Camera tab crashes the app', priority='urgent', category='technical')
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `TicketExtraction` — add/change fields; both parsing methods pick up the new schema automatically

### See also

- [`../prompt_templates/README.md`](../prompt_templates/README.md) — `get_format_instructions()` is injected via the same `.partial()` mechanism shown there
- [`../../Core_Architecture/tool_use/README.md`](../../Core_Architecture/tool_use/README.md) — the raw-SDK tool-schema enforcement `with_structured_output` builds on
