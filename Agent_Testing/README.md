# Agent_Testing

The software-testing discipline for an agent's code, as distinct from `../Benchmarking/`'s job of SCORING an agent's live output quality: unit-testing tools in isolation, integration-testing the orchestration loop against a stubbed LLM, red-teaming with an adversarial corpus, extracting shared fixtures/mocks so the suite stays fast, and layering all of it plus `../Benchmarking/`'s expensive templates into one cost-aware pipeline as the agent and its test suite both grow.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`tool_unit_testing/`](tool_unit_testing/README.md) | The base layer — testing individual tool functions in isolation, no LLM call, dependency-injected state so tests never interfere with each other |
| 2 | [`agent_integration_testing/`](agent_integration_testing/README.md) | One layer up — testing the full multi-turn, multi-tool-call orchestration LOOP against a stubbed LLM client, deterministically and for free |
| 3 | [`adversarial_safety_testing/`](adversarial_safety_testing/README.md) | A different axis entirely — a growing corpus of known attacks (and known-benign lookalikes) proving the agent refuses or degrades safely |
| 4 | [`fixture_and_mock_management/`](fixture_and_mock_management/README.md) | The infrastructure that keeps 1-3 maintainable at scale — shared, reusable fixture/mock builders instead of each test file hand-rolling its own |
| 5 | [`test_suite_pyramid/`](test_suite_pyramid/README.md) | Tying it all together — layering fast (1-3) and slow (LLM-judge/live, from `../Benchmarking/`) tiers into one pipeline, gated by cost |

## Setup

Same as the rest of the repo, plus `pytest`:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here   # only needed for test_suite_pyramid/'s optional live tier
```

Run any template from the repo root, either directly or via pytest, e.g.:

```bash
python3 Agent_Testing/tool_unit_testing/tool_unit_testing.py
pytest Agent_Testing/tool_unit_testing/tool_unit_testing.py
```

Run every fast-tier file in this topic together:

```bash
pytest Agent_Testing/tool_unit_testing/ Agent_Testing/agent_integration_testing/ Agent_Testing/adversarial_safety_testing/ Agent_Testing/fixture_and_mock_management/
```

## How these relate to each other

| | What breaks if this layer is missing | Speed / cost | Needs `ANTHROPIC_API_KEY`? |
|---|---|---|---|
| `tool_unit_testing/` | A tool's own logic (a bad amount, an unknown ID) ships broken | Milliseconds, free | No |
| `agent_integration_testing/` | Tools work individually but the LOOP calling them mis-orchestrates (wrong stop condition, lost tool results, no retry) | Milliseconds, free | No |
| `adversarial_safety_testing/` | A previously-blocked attack silently starts working again after an unrelated change | Milliseconds, free | No |
| `fixture_and_mock_management/` | Nothing breaks directly, but the suite above gets slower to extend and drifts out of sync with the real API shape as it grows | Milliseconds, free | No |
| `test_suite_pyramid/`'s slow tier | Output QUALITY regresses (an answer gets worse, less grounded, or less polite) in a way no structural test above can see | Seconds, costs money | Yes |

`tool_unit_testing/` and `agent_integration_testing/` are BEHAVIORAL correctness at two different scopes — one function, then the whole loop. `adversarial_safety_testing/` is a different question altogether: not "is this correct" but "does this hold up under attack." `fixture_and_mock_management/` isn't a test layer at all — it's the shared plumbing the first three increasingly depend on as the suite grows past what copy-pasting fixtures can sustain. `test_suite_pyramid/` is the only layer that reaches outside this topic, pulling `../Benchmarking/`'s LLM-judge and trace-evaluation templates in as the expensive top tier, and is the piece that answers the "how do we scale this as the agent keeps changing" question directly: gate the cheap tiers on every commit, gate the expensive tier on a schedule, and let every real bug found along the way become a new permanent test case in whichever tier it belongs to.
