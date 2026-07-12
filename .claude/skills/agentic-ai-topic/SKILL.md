---
name: agentic-ai-topic
description: Use whenever the user gives a topic name plus a list of subtopics (each with a short definition) and asks to create templates/examples for it in this repo — e.g. "create a directory called X, subdirectories for each of these topics: A — definition. B — definition. ...". Builds one runnable, heavily-commented Python template per subtopic in its own subdirectory, with a README per subdirectory and one for the parent topic, updates the root README index, verifies everything, and stops short of committing until asked. Also use when asked to reorganize/move existing template directories within this repo, since the same path-reference-integrity steps apply.
---

# Agentic AI topic builder

This repo (`agentic_ai_basics`) is a learning series of small, self-contained
Python templates that teach agentic-AI concepts by example, each living in
its own directory with Claude API calls and heavy explanatory comments. This
skill is the repeatable procedure for turning a topic + subtopic list into a
finished, matching addition to that repo.

**Default to running this procedure autonomously end-to-end** — plan the use
cases yourself, write the code, verify it, write the docs. Only stop to ask
the user when something is genuinely ambiguous (e.g. two equally-good real
external services to demo against). Stop and report back once everything is
written and verified; **do not run `git commit` until the user explicitly
asks for it** (they will typically say "commit" as a separate message after
reviewing).

## 0. Read before starting

Skim 2-3 recently-added topic directories (check `git log --oneline` for the
most recent `Add <Topic>: ...` commits, e.g. `Multi_Agent_Systems/`,
`Tools_and_Actions/`, `Memory/`) to recalibrate on current file/README
conventions before writing new ones — the concrete examples below capture
the pattern but the two or three most recent real directories are the
ground truth.

## 1. Parse the request into a directory plan

The user's message has the shape: a topic name, followed by N subtopics
(usually 5), each as `Subtopic Name — one-line definition.`

- **Main directory name**: convert the topic name to `Title_Case_With_Underscores`
  — capitalize each word, join with `_`, turn `&`/`/`/`-` into `and`/`_`.
  Examples from this repo: `Memory`, `RAG_and_Knowledge`,
  `Planning_and_Reasoning`, `Tools_and_Actions`, `Multi_Agent_Systems`.
- **Subdirectory names**: one per subtopic, `snake_case`, short — drop
  parenthetical abbreviations and slashes, pick the clearer half of an
  "X / Y" label. Examples: "Chain-of-Thought (CoT)" → `chain_of_thought`;
  "Sub-agent / Worker Agent" → `worker_agent`; "API Connectors / MCP" →
  `api_connectors_mcp`; "Browser / Computer Use" → `browser_computer_use`.
- **Script filename**: `<subdirectory_name>.py` inside each subdirectory
  (e.g. `orchestrator/orchestrator.py`, `swarm/swarm.py`). Exception: if the
  subtopic is a variant of an existing top-level pattern already named
  `basic_*.py` elsewhere in the repo, match that naming instead — but for a
  fresh topic directory, `<subdirectory_name>.py` is the norm.
- Create all directories in one shot: `mkdir -p Topic/sub1 Topic/sub2 ...`.

## 2. Design one concrete template per subtopic

For each subtopic, before writing code, decide:

- **A small, realistic use case** that makes the concept's mechanic
  observable in a short terminal session — not a toy that could be any
  concept. Reuse a recognizable domain the repo has used before (task
  managers, weather, support tickets, investment proposals, checkout
  flows) when it fits, rather than inventing a new one for its own sake.
- **Real vs. mocked implementation.** Prefer a real, live mechanism
  whenever one exists and is reachable:
  - A native Claude server-side tool (`code_execution_*`, `web_search_*`)
    → declare it for real, no mocking. Check the bundled `claude-api` skill
    for current tool-type strings and beta headers before writing — they
    change over time.
  - A real external API with no auth required (e.g. a free public REST
    API) → call it for real with the standard library
    (`urllib.request`), verified with a live call during testing, not
    just a mock.
  - Something requiring infrastructure this environment doesn't have
    (a real browser/display for computer use, a real MCP server) → build
    the closest runnable alternative (e.g. structured page state instead
    of pixel screenshots) and say so explicitly in the module docstring —
    don't silently pretend a mock is the real thing. For the genuinely
    unrunnable option (e.g. MCP against a real server), show it as a
    clearly-labeled commented reference block instead of executing it.
  - Otherwise, a well-labeled local mock (in-memory dict, small JSON
    file, a hand-rolled `embed()`/BM25 implementation) is the norm — most
    templates in this repo are this category, and that's fine as long as
    the comments are honest about what's simplified and why.
- **What this template contrasts with.** Every template in this repo
  cross-references at least one other existing template — grep the repo
  first (`grep -rln "<related concept>" --include="*.py" .`) for the
  closest relatives, and write the new one explicitly against that
  backdrop ("contrast with `../../X/y.py`, which does Z instead").

## 3. Write the template file

Follow the established shape (see any recent `*/**.py` file for a live
example):

```python
"""
CONCEPT: <Name> — <one-line definition, close to the user's own wording>.

<2-4 paragraphs: what this mechanic actually is, why it matters, and how
it contrasts with 1-2 specific other templates elsewhere in the repo by
relative path, e.g. ../../Core_Architecture/tool_use/basic_agentic_tools.py>

Use case: <one line>. Type 'exit' to quit/end the conversation.
"""

from __future__ import annotations  # if using `|` union types

import os
import sys
# ... other stdlib imports as needed

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096          # tune per template's output length needs
EFFORT = "medium"
SYSTEM_PROMPT = "..."      # or per-role prompts for multi-agent templates

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ... TOOLS list (JSON Schema dicts) and/or tool implementation functions,
# each with a CONCEPT comment at the specific line/block that demonstrates
# the mechanic, not just what the code does

def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    ...  # returns (result_text, is_error)

def run_turn(messages: list[dict]) -> None:
    ...  # or the pattern-specific loop shape (handoff, supervise, swarm fan-out, etc.)

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)
    print("<intro>. Type 'exit' to quit.\n")
    print("Try: \"<one realistic example prompt>\"\n")
    ...  # input loop

if __name__ == "__main__":
    main()
```

Notes:
- **Comments are deliberately generous here** — this overrides the usual
  terse-comments default. The whole point of this repo is comments that
  teach the *why*, not just the *what*. Keep the module docstring and
  inline `CONCEPT:` comments; don't strip them down.
- Any tool that touches the filesystem or evaluates expressions needs an
  explicit safety treatment (canonicalize + boundary-check paths; `ast`-based
  safe eval instead of `eval()`) with a comment explaining that tool input
  is untrusted even when it comes from the model, not the end user directly.
- Client-side custom tools follow the `TOOLS` list + `execute_tool`
  dispatch + tool-calling loop shape used throughout the repo. Server-side
  Anthropic tools (code execution, web search) have **no** `execute_tool`
  at all — say so in the docstring, since it's a common point of confusion
  coming from the client-side pattern.

## 4. Verify each file before moving to the next

Every template, no exceptions:

1. `python3 -c "import ast; ast.parse(open('X.py').read()); print('syntax OK')"`
2. Offline logic tests, run from inside the subdirectory with
   `os.environ.setdefault('ANTHROPIC_API_KEY', 'x')` set first:
   - Exercise every pure function / tool implementation directly with
     both success and failure inputs (unknown IDs, malformed input,
     boundary conditions).
   - For orchestration logic that calls `client.messages.create` (loops,
     retries, handoff state, parallel fan-out), **monkeypatch** the
     function under test or `client.messages.create` with a fake using
     `types.SimpleNamespace` shaped like the real response
     (`.content`, `.stop_reason`, block `.type`/`.text`/`.name`/`.input`),
     and assert on the resulting state transitions/call counts — don't
     just eyeball the code.
   - For real external calls (live APIs, real Claude native tools),
     actually exercise them for real when network/API access is available
     rather than only mocking — this repo has genuinely called live
     weather APIs and real Claude tool-use responses during development.
   - For concurrency claims (a swarm/parallel template), prove it with a
     timing test (N sleeping fakes should complete in ~1x the sleep, not
     ~Nx) rather than asserting it from the code shape alone.
3. Fix anything the tests catch immediately, re-run.
4. Delete any runtime-generated artifacts the test run left behind
   (`*.json` data files, a `sandbox/` dir) before moving on.

## 5. Write the per-subdirectory README

```markdown
# <subdirectory_name>

<One-line restatement of the subtopic's definition.>

## <script_name>.py

<One or two sentences: the demo scenario, how to end the session.>

### Concepts covered

- **`<function/class/constant name>`** — what it does, *and* how it
  contrasts with or builds on a specific other template (relative link).
- (4-6 bullets total, each anchored to a concrete named thing in the code)

### Run

From the repo root:

​```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Topic/subdirectory/script.py
​```

Try:

​```
<a realistic example transcript, including tool-call/observation lines
where relevant>
​```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- <any subtopic-specific constants>

### See also

- `../other_subdir/README.md` — one-line reason it's related
- `../../other_topic/thing.py` — one-line reason it's related
```

## 6. Write the parent topic README

```markdown
# Topic_Name

<1-2 sentences: what unifies the subtopics, what differentiates them.>

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`sub1/`](sub1/README.md) | ... |
...

## Setup

Same as the rest of the repo:

​```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
​```

Run any template from the repo root, e.g.:

​```bash
python3 Topic_Name/sub1/sub1.py
​```

## How these relate to each other

<A comparison table along whatever axes actually differentiate this
topic's subtopics — e.g. scope/persistence/structure/who-manages-it for
Memory; control/coordination/validation for Multi-Agent Systems;
server-side-vs-client-side for Tools_and_Actions. Pick real axes, don't
force a generic table.>
```

## 7. Update the root README index — and only the index

Add exactly **one** new row to the `## Templates` table in the root
`README.md`:

```
| [`Topic_Name/`](Topic_Name/README.md) | <one-line combined description covering all subtopics> | `sub1.py`, `sub2.py`, ... |
```

The root README must stay a thin index (setup instructions + this one
table) — never add subtopic-level detail there. If this task is a
reorganization that removes a standalone top-level directory (folding it
into the new topic dir), delete its old row too.

## 8. If moving/reorganizing existing directories

When the ask is "move X into Y" rather than "create a new topic":

1. `git mv old_path new_path` for each moved directory (preserves history
   as a rename).
2. Before editing anything, grep for every reference to the moved path,
   **anchored with a trailing slash** to avoid false positives from
   unrelated identifiers that happen to contain the same substring (e.g.
   `tool_use_id` matching a grep for `tool_use`):
   `grep -rn "old_dir/" --include="*.py" --include="*.md" .`
3. Classify each hit by its new relative depth and fix precisely:
   - The moved file's own references to *other* unmoved directories need
     one more `../` (it's now nested one level deeper).
   - The moved file's references to files that moved *with* it (siblings)
     stay unchanged.
   - External files' references *to* the moved path need the new
     directory inserted into the relative path.
   - Files that are now direct siblings of the moved dir (inside the same
     new parent) can *drop* a `../` level.
4. Re-run the full verification pass (§4) after the path fixes, plus a
   repo-wide `ast.parse` sweep, plus a repeat of the anchored grep to
   confirm no stray old-path references remain.
5. Update the root README and the new parent topic README's tables
   accordingly (§6, §7).

## 9. `.gitignore` hygiene

Any new runtime-generated file a template creates (a `*.json` data store,
a `sandbox/` directory) gets added to `.gitignore` in the same batch —
check the existing entries first so the pattern is consistent, and don't
leave test-run artifacts sitting untracked when you're done (§4 already
covers cleanup, but double check with `git status` before the final
report).

## 10. Final report — do not commit yet

Before ending the turn:
- Run one last repo-wide `ast.parse` sweep and `git status` to confirm
  only the intended files are new/modified.
- Summarize what was created (topic dir, N subdirectories, N templates,
  what each demonstrates and what got verified) concisely — this repo's
  history shows the user reviews before asking to commit, so don't
  preemptively run `git add`/`git commit`.
- If asked to commit, stage with `git add -A`, sanity-check `git status`
  matches expectations, write a commit message in the imperative
  ("Add Topic: sub1, sub2, ..."), body as a bullet list — one bullet per
  template naming what it demonstrates and what verification caught/proved
  — end with a blank line and `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  (match the model name actually in use), then `git commit`, then confirm
  with `git log --oneline` and `git status`.
