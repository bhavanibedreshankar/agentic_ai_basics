# multi_model_fallback

Failing over to a different model when the primary one is transiently unavailable, instead of failing the whole request.

## multi_model_fallback.py

A chat assistant with a `FALLBACK_CHAIN` of `["claude-sonnet-5", "claude-haiku-4-5-20251001"]`. If the primary model raises a retryable error (overload, rate limit, transient server/connection issue), the same question is retried against the next model in the chain. Type `exit` to quit.

### Concepts covered

- **`RETRYABLE_ERRORS`** — a deliberately narrow tuple: `OverloadedError`, `RateLimitError`, `InternalServerError`, `APIConnectionError`, `APITimeoutError`. `BadRequestError` and `AuthenticationError` are deliberately excluded — those fail identically on every model in the chain, so falling back would only mask a real bug instead of recovering from one.
- **`ask_with_fallback`** — the retry loop: catches only `RETRYABLE_ERRORS`, tries the next model in `FALLBACK_CHAIN` on failure, and re-raises the last error if every model fails.
- **Reliability routing vs. every other template in this topic** — `../task_classifier_router/README.md` and `../cost_aware_model_selection/README.md` route based on what the *request* needs; `../complexity_based_escalation/README.md` routes based on how confident the *answer* was. This template routes based purely on whether the call *succeeded at all* — it doesn't inspect the request or the response content, only whether the primary model was reachable.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Model_Routing/multi_model_fallback/multi_model_fallback.py
```

Under normal conditions every request just answers from the primary model — the fallback path only shows up during a real outage, which is awkward to demonstrate live. The template's automated tests instead construct real `anthropic.OverloadedError` / `anthropic.RateLimitError` / `anthropic.BadRequestError` instances (via a minimal `httpx.Response`) and monkeypatch `client.messages.create` to raise them, proving: (1) retryable errors correctly trigger a fallback to the next model, (2) non-retryable errors propagate without ever trying the fallback, and (3) exhausting the whole chain raises a clear `RuntimeError` naming the last error.

### Configuration

- `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `FALLBACK_CHAIN` — models to try, in priority order
- `RETRYABLE_ERRORS` — which exception types are worth failing over for

### See also

- `../complexity_based_escalation/README.md` — a superficially similar "try model A, then model B" shape, but triggered by low self-reported confidence rather than an API error
- `../../Safety_and_Control/README.md` — other templates about keeping an agent's behavior bounded and predictable under failure
