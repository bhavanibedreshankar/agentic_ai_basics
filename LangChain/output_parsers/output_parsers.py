"""
CONCEPT: Output Parsers — turning a model's raw text into a validated,
typed Python object instead of a string you have to parse by hand.

Every raw-`anthropic`-SDK template in this repo that needs STRUCTURED
input already gets it via tool use — e.g. ../../Core_Architecture/tool_use/basic_agentic_tools.py's
`tool_input` arrives pre-validated against a JSON Schema because the model
is filling in TOOL ARGUMENTS, and the API enforces the schema before
Claude can even return them. Output parsers solve the mirror-image
problem: getting structured data out of what the model would otherwise
return as its final, free-form ANSWER, with no tool call involved at all.

This template shows two different ways LangChain does that, in increasing
order of reliability:
  1. `PydanticOutputParser` — asks the model, via plain text instructions
     appended to the prompt (`parser.get_format_instructions()`), to
     format its answer as JSON matching a schema, then parses that text
     into a Pydantic model. This can still fail if the model's text
     doesn't quite match — it's a convention the model is asked to follow,
     not one the API enforces.
  2. `llm.with_structured_output(Model)` — wraps the model so the SAME
     schema is enforced the way basic_agentic_tools.py's tool schemas are:
     under the hood, it binds the Pydantic model as a tool the model must
     call, so the arguments arrive pre-validated instead of being parsed
     out of free text after the fact. No format instructions needed in
     the prompt at all.

Use case: extracting a structured `TicketExtraction` (summary, priority,
category) from the same free-form support ticket text used in
../prompt_templates/prompt_templates.py, once via each method, so the
reliability difference in the docstring above is something you can
actually see the code for, not just take on faith. Type 'exit' to end the
conversation.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024


class TicketExtraction(BaseModel):
    """The schema both parsing methods below target — one Pydantic model,
    used two different ways."""

    summary: str = Field(description="One-sentence summary of the ticket")
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal["billing", "technical", "account"]


# ---------------------------------------------------------------------------
# METHOD 1: PydanticOutputParser — text-instruction-based
# ---------------------------------------------------------------------------
_pydantic_parser = PydanticOutputParser(pydantic_object=TicketExtraction)

# CONCEPT: get_format_instructions() generates the literal text block
# telling the model what JSON shape to produce — this is what makes the
# schema enforcement "soft": it's instructions IN the prompt, not a
# constraint the API applies to the response.
TEXT_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract structured information from the support ticket.\n{format_instructions}"),
        ("human", "{ticket_text}"),
    ]
).partial(format_instructions=_pydantic_parser.get_format_instructions())


def build_text_parser_chain(llm: BaseChatModel) -> Runnable:
    # CONCEPT: the parser slots into the pipeline exactly like
    # StrOutputParser does in ../chains/chains.py — it's just a Runnable
    # whose job is "turn the AIMessage into something else", here a
    # TicketExtraction instead of a plain string.
    return TEXT_PARSE_PROMPT | llm | _pydantic_parser


# ---------------------------------------------------------------------------
# METHOD 2: with_structured_output — tool-calling-backed, schema-enforced
# ---------------------------------------------------------------------------
STRUCTURED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract structured information from the support ticket."),
        ("human", "{ticket_text}"),
    ]
)


def build_structured_output_chain(llm: BaseChatModel) -> Runnable:
    # CONCEPT: no format instructions in the prompt at all — the schema is
    # enforced by binding TicketExtraction as a tool the model must call,
    # the same mechanism basic_agentic_tools.py's TOOLS list uses, just
    # generated from the Pydantic model instead of a hand-written dict.
    structured_llm = llm.with_structured_output(TicketExtraction)
    return STRUCTURED_PROMPT | structured_llm


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    text_parser_chain = build_text_parser_chain(llm)
    structured_chain = build_structured_output_chain(llm)

    print("Ticket extraction (output parsers demo). Type 'exit' to quit.\n")
    print("Try: \"App crashes every time I open the camera tab, please fix urgently\"\n")

    while True:
        ticket_text = input("Ticket text: ").strip()
        if ticket_text.lower() == "exit":
            print("Goodbye!")
            break
        if not ticket_text:
            continue

        text_result = text_parser_chain.invoke({"ticket_text": ticket_text})
        print(f"\n[PydanticOutputParser] {text_result!r}")

        structured_result = structured_chain.invoke({"ticket_text": ticket_text})
        print(f"[with_structured_output] {structured_result!r}\n")


if __name__ == "__main__":
    main()
