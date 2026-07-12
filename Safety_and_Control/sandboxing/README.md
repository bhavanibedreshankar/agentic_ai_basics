# sandboxing

Sandboxing — isolating agent execution, especially code, to prevent unintended side effects on the host system.

## sandboxing.py

A coding assistant with a `run_command` tool, sandboxed with four independent, stacked layers: an allowlist, no shell interpretation, a timeout, and a confined working directory. Type `exit` to quit.

### Concepts covered

- **A bigger attack surface than a path check** — contrast with `../../Tools_and_Actions/file_io_tools/file_io_tools.py`, which validates one argument (a path) before a read/write. Arbitrary command execution has to defend against shell metacharacters, chaining, and substitution too — a single path check doesn't cover any of that.
- **Self-hosted, unlike `../../Tools_and_Actions/code_interpreter/code_interpreter.py`** — that template hands execution to Anthropic's own hosted sandbox entirely; there's no sandboxing code in that file because the infrastructure already does it. This template is what you'd build if you're hosting the execution environment yourself.
- **Four stacked layers, each closing a different attack class**: an allowlist checked on the parsed program name (layer 1), `shell=False` with `shlex.split` so metacharacters are inert argument text rather than shell syntax (layer 2), a wall-clock timeout (layer 3), and a `cwd` pinned to a sandbox directory (layer 4).
- **Verified with real subprocess execution, not mocks** — since this mechanism never touches the API, every layer was proven directly: a `;`-chained command prints its chain as literal text instead of running the second command; `$(whoami)` prints literally instead of being substituted; a simulated hang is killed at the configured timeout; a file written into the sandbox directory is found via a relative path, proving `cwd` is genuinely confined there.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Safety_and_Control/sandboxing/sandboxing.py
```

Try:

```
You: Run rm -rf /
  [tool] run_command({'command': 'rm -rf /'})
  [result] Sandbox rejected this command: 'rm' is not an allowed command. Allowed: ['cat', 'date', 'echo', 'ls', 'pwd', 'wc']
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `ALLOWED_COMMANDS` — the command allowlist (default: `echo`, `ls`, `cat`, `wc`, `date`, `pwd`)
- `COMMAND_TIMEOUT_SECONDS` — the wall-clock limit per command (default: `5`)
- `SANDBOX_DIR` — the confined working directory (default: `sandbox/` next to the script)

### See also

- `../../Tools_and_Actions/file_io_tools/README.md` — the narrower, path-only version of this same "don't trust model-supplied input" principle
- `../../Tools_and_Actions/code_interpreter/README.md` — the fully-hosted alternative where Anthropic runs the sandbox instead of you
