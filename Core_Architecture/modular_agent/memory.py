"""
CONCEPT: Memory — the agent's growing record of what's happened, kept
separate from both the loop that reads/writes it and the wire format the
LLM API expects.

In ../agent/agent.py, conversation history is a bare `list[dict]` built and
mutated directly inside pursue_goal(). That works, but it means anything
that wants to change how history is stored (cap its length, persist it,
tag entries by kind, swap in a smarter memory strategy — see
../../Memory/ for real examples of each) has to edit the loop itself.

Here, Memory is its own object with its own API (`add`, `get_memories`,
`as_message_list`). loop.py never touches `_items` directly — it only calls
these methods. That means you can later replace this whole class with, say,
a version that summarizes old turns or persists to disk (à la
../../Memory/working_memory.py or ../../Memory/external_memory.py),
and nothing in loop.py, agent.py, or language.py has to change, as long as
the new class keeps the same three methods.

Each MemoryItem stores its content in the shape the Anthropic Messages API
expects for that role (a string for simple text, a list of content blocks
for assistant turns and tool results) — see language.py's docstring for why
that's a deliberate shortcut rather than full provider-agnostic storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MemoryItem:
    role: str  # "user" or "assistant" — must match the roles the LLM API accepts
    content: Any  # str, or a list of content blocks (tool_use / tool_result / text)
    kind: str = "message"  # free-form tag for introspection/filtering, e.g. "user_input", "action_result"


@dataclass
class Memory:
    """An append-only, queryable record of the conversation so far."""

    _items: List[MemoryItem] = field(default_factory=list)

    def add(self, role: str, content: Any, kind: str = "message") -> None:
        self._items.append(MemoryItem(role=role, content=content, kind=kind))

    def get_memories(self, kind: Optional[str] = None) -> List[MemoryItem]:
        """Return all items, or only those tagged with a given `kind`.

        Useful for features that need to *inspect* history without owning the
        loop — e.g. a benchmarking script counting action_result items, or a
        future pruning feature deciding what's safe to drop.
        """
        if kind is None:
            return list(self._items)
        return [item for item in self._items if item.kind == kind]

    def as_message_list(self) -> List[dict]:
        """Render stored items into the `messages` list the LLM API call expects."""
        return [{"role": item.role, "content": item.content} for item in self._items]
