# dynamic_prompt_construction

Building a prompt's actual content at runtime from current variables and state, instead of sending one fixed, hardcoded string every time.

## dynamic_prompt_construction.py

A support assistant whose system prompt is rebuilt from scratch every turn out of independent, conditionally-included sections: a fixed persona, a tier-based service block, an urgency block (only when the message reads urgent), a policy-facts block (only when a cheap keyword match finds something relevant), and a stall block (only after several turns without resolving). Each constructed prompt is printed before the response, so the assembly is visible. Set an account tier at the start, then type `exit` to end the conversation.

### Concepts covered

- **`build_sections` / `build_system_prompt`** — the assembly itself: each of five functions computes independently, and only non-empty sections get joined in — the STRUCTURE of the final prompt (how many sections, which ones) varies call to call, not just values filled into a fixed shape.
- **`urgency_instructions`, `policy_instructions`** — sections driven by the CURRENT message's content, via cheap keyword checks rather than the embedding-based retrieval in `../../09_RAG_and_Knowledge/`; this is the point — dynamic prompt construction doesn't require a full RAG stack.
- **`stall_instructions`** — a section driven by CONVERSATION state (turn count) rather than message content at all, so the exact same message produces a different prompt on turn 1 than on turn 4.
- Contrast with `../../01_Core_Architecture/system_prompt/system_prompt.py`: that file picks ONE complete, pre-written prompt from a fixed set; this file assembles ONE prompt from independently-decided pieces.
- Contrast with `../../17_LangChain/prompt_templates/prompt_templates.py`: that file fills slots in one fixed template STRUCTURE; here the structure itself changes based on runtime conditions.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 08_Agent_Frameworks_and_Patterns/dynamic_prompt_construction/dynamic_prompt_construction.py
```

Try:

```
Account tier ('free' or 'premium'): free
You: How do I cancel my subscription?
  [constructed system prompt, 3 section(s) included]
  ...

You: This is urgent, I need a refund ASAP!
  [constructed system prompt, 4 section(s) included]
  ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../01_Core_Architecture/basics/README.md`
- `URGENCY_KEYWORDS`, `POLICY_KEYWORDS`, `POLICIES` — the heuristics and facts driving which optional sections appear
- `STALL_THRESHOLD` — how many turns before the human-escalation section appears

### See also

- `../../01_Core_Architecture/system_prompt/README.md` — choosing between complete, pre-written prompts, rather than assembling one from pieces
- `../../17_LangChain/prompt_templates/README.md` — LangChain's slot-filling take on a related but structurally different idea
- `../../15_Self_Evolving_Agents/self_evolving_agents/README.md` — a system prompt that also changes at runtime, but by loading rules learned from past feedback and persisted across sessions, not rebuilt fresh from this-turn state
- `../../09_RAG_and_Knowledge/rag/README.md` — real embedding-based retrieval, contrasted with this file's deliberately cheap keyword lookup
