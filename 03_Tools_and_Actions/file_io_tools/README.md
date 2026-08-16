# file_io_tools

File I/O Tools — read, write, and edit files on disk, the core capability behind every coding agent.

## file_io_tools.py

A coding assistant using the real Anthropic-defined text editor tool (`text_editor_20250728`), sandboxed to a `sandbox/` directory next to this script. Type `exit` to end the conversation.

### Concepts covered

- **Anthropic-defined, but client-executed** — unlike `../code_interpreter/` and `../web_search/`, this tool's schema is built into the model (declare it by `type`/`name` only, no `input_schema`), but *your code* still has to perform the actual file operation and send a result back — `execute_text_editor` and the `run_turn` loop are both present here, unlike the two purely server-side templates.
- **The security boundary is the point of this file** — `path` in every command is model-supplied, and therefore untrusted, even though it comes from Claude rather than the end user directly (a manipulated tool result or document Claude was asked to read could smuggle in a bad path). `_resolve_safe_path` canonicalizes every path and verifies it's still inside `SANDBOX_ROOT` before any read or write happens — verified in testing against both `..`-traversal attempts and absolute paths.
- **Absolute paths are neutralized, not just rejected** — a model-supplied `/etc/passwd` is deliberately remapped to `sandbox/etc/passwd` (a leading `/` is treated as the sandbox's own root), so it fails as "not found" rather than needing special-case detection as an attack. Actual `..` traversal is what gets rejected outright as a security error.
- **The four real commands** — `view` (a file or a directory listing), `create`, `str_replace` (requires the target string to match exactly once), and `insert`. `undo_edit` from older tool versions is intentionally not implemented — it's no longer supported by this tool version.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 03_Tools_and_Actions/file_io_tools/file_io_tools.py
```

Try:

```
You: Create a file called notes.txt with a short haiku about autumn, then show me its contents.

  [tool] str_replace_based_edit_tool({'command': 'create', 'path': 'notes.txt', 'file_text': '...'})
  [result] Created notes.txt
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../01_Core_Architecture/basics/README.md`
- `SANDBOX_ROOT` — the directory every file operation is confined to (default: `sandbox/` next to the script)

### See also

- `../code_interpreter/README.md`, `../web_search/README.md` — the two server-side tools this template's client-side dispatch loop contrasts with
- `shared/tool-use-concepts.md` § Client-Side Tools (in the `claude-api` skill) — the general path-traversal rule this template implements
