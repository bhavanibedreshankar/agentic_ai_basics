"""
CONCEPT: Context (prompt) caching — telling the API "everything up to
this point is going to reappear byte-for-byte in a future request, store
a copy of how you processed it" so a later call re-reads that prefix at
roughly 1/10th the normal input price instead of reprocessing it from
scratch.

This is a SERVER-SIDE mechanism (a `cache_control` marker on a content
block), not a client-side data structure — contrast with
../tool_result_caching/tool_result_caching.py, which memoizes function
RESULTS in a plain Python dict on our side of the wire. Prompt caching
instead caches the model's own internal read of a prompt PREFIX (system
prompt, tool definitions, early conversation turns) inside Anthropic's
infrastructure. ../../Core_Architecture/token_tracking/basic_token_tracking.py
already prints the `cache_creation_input_tokens` / `cache_read_input_tokens`
fields on every response, but never actually sets a `cache_control`
marker — those fields sit at 0 the whole time. This template is the
other half: actually opting in, and watching those fields turn non-zero.

The mental model is a PREFIX MATCH. The cache key is the exact rendered
bytes of the prompt up to a `cache_control` breakpoint (render order is
tools -> system -> messages). Change one byte anywhere in that prefix —
a timestamp, a reordered key, a different tool — and everything from
that point on misses. So the #1 rule for cacheable prompts is: keep the
stable part stable. This template's system prompt is a large, fixed
support-policy document that never changes between turns, which is
exactly the shape that benefits: pay the full price ONCE to write it to
cache, then read it back cheaply on every subsequent turn instead of
reprocessing thousands of tokens of policy text again and again.

Use case: a customer-support agent whose system prompt is a lengthy,
unchanging policy handbook (too big to want to reprocess at full price
every turn) plus a small tool. The FIRST turn pays a cache-write premium
(~1.25x normal input price) to store that handbook. Every turn after
that reads it back at ~0.1x price instead of ~1x — and because the
breakpoint also moves forward each turn (see add_cache_breakpoint), the
growing conversation history gets folded into the cached prefix too, so
the cache-read share keeps growing as the chat goes on.

Type 'exit' to end the conversation and see a cache economics summary.
"""

from __future__ import annotations

import copy
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the stable prefix. This has to be big enough to actually be
# worth caching -- and, more importantly, big enough to clear the
# MINIMUM CACHEABLE PREFIX for the model in use. That minimum is
# model-dependent (1024 tokens for Sonnet, 2048 for several older/smaller
# models, 4096 for Opus/Haiku 4.5) and there's no error if you're under
# it -- the marker is silently ignored and cache_creation_input_tokens
# comes back 0. This handbook runs a few thousand tokens, comfortably
# over every threshold, specifically so the cache fields below aren't
# zero by accident.
#
# Notice what's NOT in here: no `datetime.now()`, no session/user ID, no
# random request ID. Interpolating any of those into a "stable" prompt
# is the single most common way to silently break caching -- the prefix
# looks the same to a human reading the code, but the rendered bytes
# differ on every request, so every request misses.
# ---------------------------------------------------------------------------
POLICY_HANDBOOK = """You are a support agent for Nimbus Cloud Storage. Answer customer
questions using ONLY the policies below. If something isn't covered, say so plainly
rather than guessing.

SECTION 1 - ACCOUNT TIERS
Nimbus offers three tiers: Free (5GB, single device, community support only),
Plus ($6/mo, 500GB, up to 3 devices, email support with a 48-hour response target),
and Business ($20/mo per seat, 2TB pooled per seat, unlimited devices, priority
support with a 4-hour response target during business hours, plus a dedicated
account manager for accounts with 20+ seats). Tier changes take effect immediately
for upgrades and at the end of the current billing cycle for downgrades. Storage
over a new, lower tier's limit is not deleted; uploads are simply blocked until the
account is back under the limit or the customer upgrades again.

SECTION 2 - REFUND POLICY
Monthly subscriptions are refundable within 14 days of the charge if the customer
has used less than 10% of their storage quota in that period. Annual subscriptions
are refundable within 30 days under the same usage condition. Refunds are never
issued for the current partial month if a customer downgrades or cancels; they keep
access through the end of the period they already paid for. Refund requests outside
these windows may still be escalated to a supervisor at the agent's discretion if
there's a clear billing error on Nimbus's side (double charge, wrong tier charged).

SECTION 3 - DATA RETENTION AND DELETION
Deleted files move to a Trash folder and are permanently purged 30 days after
deletion, or immediately if the customer empties Trash manually. Cancelling a paid
plan does not delete data; the account reverts to Free-tier limits and any storage
over 5GB becomes read-only (downloadable but not addable-to) for 90 days, after
which it is purged if the account hasn't been upgraded again. There is no way to
recover data after the 90-day read-only window closes, even for a supervisor --
agents must be explicit about this timeline rather than implying recovery is always
possible.

SECTION 4 - SECURITY AND ACCESS
All files are encrypted at rest (AES-256) and in transit (TLS 1.3). Two-factor
authentication is available on all tiers and required on Business. Support agents
can NEVER ask for or accept a customer's password, recovery code, or 2FA code over
chat or email under any circumstance -- account recovery always goes through the
identity-verification flow, which checks billing details and a device history
fingerprint instead. Suspicious login alerts are sent automatically when a login
occurs from a new device or country; customers cannot opt out of these alerts.

SECTION 5 - SHARING AND COLLABORATION
Free-tier customers can share individual files via a public link (view-only,
optionally password-protected, no expiration control). Plus adds folder sharing
and link expiration dates. Business adds granular per-user permissions (view,
comment, edit) and org-wide sharing policies an admin can enforce (e.g. disallow
public links entirely). Shared links inherit the file owner's tier limits, not the
recipient's -- a Free-tier owner's shared folder is still capped at 5GB even if
everyone viewing it is on Business.

SECTION 6 - SLA AND UPTIME CREDITS
Business accounts have a 99.9% monthly uptime SLA. If uptime falls below that in
a calendar month, affected accounts are credited 5% of that month's fee per
additional 0.1% of downtime, capped at 50% of the monthly fee. Credits are applied
automatically to the next invoice; they are not paid out as cash and cannot be
transferred between accounts. Free and Plus tiers carry no formal SLA, though
extended outages are still communicated via the status page.

SECTION 7 - ACCEPTABLE USE
Storing content that infringes copyright, contains malware, or violates local law
in the customer's jurisdiction is prohibited on all tiers. Nimbus does not scan file
contents proactively for anything other than known malware signatures, but will act
on valid takedown notices and law-enforcement requests. A first violation gets a
warning and the offending file quarantined (owner-visible but not shareable);
repeat violations can result in account suspension. Agents should never promise a
specific enforcement action beyond what's documented here -- escalate ambiguous
cases rather than guessing.

SECTION 8 - BILLING DISPUTES
Disputed charges are investigated within 5 business days. While a dispute is open,
the account keeps its current tier's access (it is not downgraded pending
investigation). If Nimbus's investigation confirms a billing error, the customer is
refunded the disputed amount plus one month of service credit. If the charge is
confirmed correct, the agent should explain the specific line item rather than
simply asserting the charge is valid.

SECTION 9 - MIGRATIONS AND EXPORTS
Customers can export their entire library as a set of dated ZIP archives at any
time, on any tier, with no throughput cap beyond their plan's normal transfer
limits. Business accounts additionally get a direct-to-S3 or direct-to-GCS bulk
export option that skips the ZIP step entirely, useful for accounts above 500GB
where a single archive would be unwieldy. Imports from other providers (Google
Drive, Dropbox, OneDrive) are supported via a one-time migration wizard; imported
files count against the destination tier's quota immediately, so agents should
confirm the customer has room before kicking off a large migration.

SECTION 10 - MOBILE AND OFFLINE SYNC
The mobile app supports offline access to any file explicitly marked "available
offline"; unmarked files are streamed on demand and not cached to the device.
Conflicts between an offline edit and a newer server version are resolved by
keeping BOTH copies (the server version, and a "(conflicted copy)" suffixed local
version) rather than silently discarding either one. Sync pauses automatically on
metered mobile data unless the customer opts in to unmetered sync in settings;
agents should check this setting first when a customer reports "my phone isn't
syncing" over cellular.
"""

# ---------------------------------------------------------------------------
# CONCEPT: a small tool, included so this template also shows the render
# order (tools -> system -> messages) that the caching docs describe.
# A cache_control marker on the LAST system block caches BOTH the tools
# above it and the system text together -- there's no need for a
# separate marker on the tool definitions.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "look_up_account_tier",
        "description": "Look up a customer's current Nimbus account tier and storage usage by account ID.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
]

# A tiny mock "database" so look_up_account_tier has something real to return.
_ACCOUNTS = {
    "acct_1": {"tier": "Plus", "used_gb": 210, "limit_gb": 500},
    "acct_2": {"tier": "Business", "used_gb": 1800, "limit_gb": 2000},
}


def look_up_account_tier(account_id: str) -> str:
    record = _ACCOUNTS.get(account_id)
    if record is None:
        return f"No account found with ID '{account_id}'."
    return (
        f"Account {account_id}: {record['tier']} tier, "
        f"{record['used_gb']}GB used of {record['limit_gb']}GB."
    )


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "look_up_account_tier":
        return look_up_account_tier(tool_input["account_id"])
    return f"Unknown tool: {name}"


def build_system_blocks() -> list[dict]:
    """CONCEPT: cache_control placement, pattern 1 -- 'large system prompt
    shared across many requests'. The marker goes on the LAST (and here,
    only) system content block. Every request in this session sends this
    exact same block, so after the first call, every later call reads it
    from cache instead of paying full price for the handbook again.
    """
    return [
        {
            "type": "text",
            "text": POLICY_HANDBOOK,
            "cache_control": {"type": "ephemeral"},  # 5-minute TTL (default)
        }
    ]


def add_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """CONCEPT: cache_control placement, pattern 2 -- 'multi-turn
    conversations'. Marking the last content block of the most recently
    appended turn means each new request reuses the ENTIRE prior
    conversation prefix (system + all earlier turns), not just the system
    prompt. Earlier breakpoints don't need to stick around -- a new,
    later breakpoint covers everything before it too, as long as those
    bytes are unchanged and within the 20-block lookback window.

    Returns a deep copy so the caller's stored `messages` list stays free
    of cache_control markers (they're only needed on the wire, and
    re-adding them fresh each call is simpler than tracking which one is
    "active").
    """
    wire_messages = copy.deepcopy(messages)
    last_block = wire_messages[-1]["content"][-1]
    last_block["cache_control"] = {"type": "ephemeral"}
    return wire_messages


class CacheStats:
    """CONCEPT: cache economics. `usage` reports three numbers that only
    mean something together: `input_tokens` (processed at full price),
    `cache_creation_input_tokens` (written this turn, ~1.25x price), and
    `cache_read_input_tokens` (served from cache, ~0.1x price). This class
    accumulates all three and estimates what the session would have cost
    WITHOUT caching, for a direct before/after comparison -- the same
    total-token idea as ../../Core_Architecture/token_tracking/basic_token_tracking.py's
    SessionUsage, but focused on the cache split rather than plain cost.
    """

    PRICE_PER_MILLION_INPUT = 3.00  # approximate Sonnet input rate; check platform.claude.com/pricing

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def add(self, usage) -> None:
        self.input_tokens += usage.input_tokens
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_input_tokens += usage.cache_read_input_tokens or 0
        self.turns += 1

    def summary(self) -> str:
        actual_cost = (
            self.input_tokens * 1.0
            + self.cache_creation_input_tokens * 1.25
            + self.cache_read_input_tokens * 0.1
        ) * (self.PRICE_PER_MILLION_INPUT / 1_000_000)
        # What the SAME total prefix tokens would have cost at full price
        # every turn, had none of it ever been cached.
        uncached_equivalent_tokens = (
            self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
        )
        uncached_cost = uncached_equivalent_tokens * (self.PRICE_PER_MILLION_INPUT / 1_000_000)
        savings_pct = (1 - actual_cost / uncached_cost) * 100 if uncached_cost else 0
        return (
            f"Turns:                   {self.turns}\n"
            f"Full-price input tokens: {self.input_tokens:,}\n"
            f"Cache writes (~1.25x):   {self.cache_creation_input_tokens:,}\n"
            f"Cache reads (~0.1x):     {self.cache_read_input_tokens:,}\n"
            f"Actual input cost:       ${actual_cost:.4f}\n"
            f"Cost with no caching:    ${uncached_cost:.4f}\n"
            f"Savings from caching:    {savings_pct:.1f}%"
        )


def run_turn(messages: list[dict], stats: CacheStats) -> str:
    wire_messages = add_cache_breakpoint(messages)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_blocks(),
        tools=TOOLS,
        output_config={"effort": EFFORT},
        messages=wire_messages,
    )

    # Keep handling tool calls until Claude produces a final text answer.
    while response.stop_reason == "tool_use":
        stats.add(response.usage)
        tool_results = []
        assistant_blocks = [b.model_dump() for b in response.content]
        for block in response.content:
            if block.type == "tool_use":
                result_text = execute_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_results})
        wire_messages = add_cache_breakpoint(messages)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_blocks(),
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=wire_messages,
        )

    stats.add(response.usage)
    reply = "".join(block.text for block in response.content if block.type == "text")
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
    print(
        f"  [cache: {response.usage.cache_read_input_tokens or 0} read / "
        f"{response.usage.cache_creation_input_tokens or 0} written / "
        f"{response.usage.input_tokens} at full price]"
    )
    return reply


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Nimbus support agent. Type 'exit' to end and see a cache economics summary.\n")
    print("Try: \"What's your refund policy?\" then a follow-up like \"What about annual plans?\"")
    print("The first turn pays a cache WRITE for the policy handbook; later turns should show cache READS.\n")

    messages: list[dict] = []
    stats = CacheStats()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("\n--- Cache economics summary ---")
            print(stats.summary())
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
        reply = run_turn(messages, stats)
        print(f"\nClaude: {reply}\n")


if __name__ == "__main__":
    main()
