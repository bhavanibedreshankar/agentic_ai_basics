# semantic_memory

Semantic memory — persistent knowledge about the world, domain facts, or user preferences. Timeless statements, not events.

## semantic_memory.py

An assistant with a structured, categorized, updatable fact store — closer to a user profile than a list. Type `exit` to end the conversation — memory persists in `semantic_memory.json`.

### Concepts covered

- **Structured and updatable, not flat and append-only** — `../memory_management/basic_agentic_memory.py` stores facts as a flat list of free-text strings that only ever grows. This template organizes facts by `category` and `key` (`remember(category, key, value)`), and — critically — a second `remember` call with the same category/key **overwrites** the old value instead of adding a contradictory second entry.
- **`forget`** — explicit removal of a fact that's no longer true, with automatic cleanup of an emptied category.
- **`recall`** — read back a specific category, or the whole profile, as structured JSON rather than a wall of prose.
- **Contrast with `../episodic_memory/`** — semantic memory holds what's currently true; episodic memory holds what happened and when. This template deliberately has no timestamps or event history — that's the other file's job.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/semantic_memory/semantic_memory.py
```

Try:

```
You: My favorite color is blue.
  [tool] remember({'category': 'preferences', 'key': 'favorite_color', 'value': 'blue'})
  [result] Remembered preferences.favorite_color = 'blue'

You: Actually, it's green now.
  [tool] remember({'category': 'preferences', 'key': 'favorite_color', 'value': 'green'})
  [result] Updated preferences.favorite_color: 'blue' -> 'green'
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `MEMORY_FILE` — where facts are persisted (default: `semantic_memory.json` next to the script)

### See also

- `../memory_management/basic_agentic_memory.py` — the simpler flat-list version this template's categorized, updatable store improves on
- `../external_memory/README.md` — a similarity-retrieved alternative for facts too numerous or unstructured to organize by category/key
