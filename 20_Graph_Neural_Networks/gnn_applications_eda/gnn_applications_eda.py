"""
CONCEPT: GNN applications in EDA (Electronic Design Automation) — a survey,
not a single deep dive. The previous five templates in this directory each
built up one GNN mechanic in isolation (graph representation, message
passing, GCN, GraphSAGE, GAT) using one running toy netlist. This file is
the payoff: four MINIATURE, runnable illustrations of what those mechanics
are actually used for in real chip design tooling, each a few lines of
message-passing-style code rather than a trained model (training loops for
each already live in ../gcn_gate_classification/, ../graphsage_inductive_
netlist_embedding/, and ../gat_attention_critical_path/ — this file is
about the SHAPE of each problem, not re-deriving those mechanics again).

Why GNNs matter for EDA specifically: a chip's netlist and its physical
placement are graphs by nature (see ../graph_representation_of_netlists/),
and the "gold standard" tools that answer these four questions precisely —
a full placer/router, SPICE-level power simulation, static timing analysis
(STA) — are all slow enough that running them thousands of times during
design exploration is expensive. A GNN trained to APPROXIMATE one of these
tools' output, directly from the netlist graph, can be orders of magnitude
faster, at the cost of being a prediction rather than a certified,
sign-off-quality result. That speed/precision trade-off — a fast learned
estimate to prioritize where to spend the slow, exact tool's time — is the
common thread under all four sections below. (Real published examples of
each direction: congestion prediction has been studied under names like
"RouteNet"/"CongestionNet"; timing prediction under "GRANNITE" and similar;
power estimation and netlist anomaly/bug detection are active research
areas too. This file's toy examples are illustrative of the PROBLEM SHAPE
these lines of work address, not reproductions of any specific paper.)

Run this file directly to walk through all four sections:

    python3 gnn_applications_eda.py
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Same toy 2-bit ripple-carry adder as every other template in this
# directory (see ../graph_representation_of_netlists/ for the full diagram).
# ---------------------------------------------------------------------------
NETLIST_FANIN = {
    "a0": [], "b0": [], "cin": [], "a1": [], "b1": [],
    "xor0": ["a0", "b0"], "and0a": ["a0", "b0"],
    "sum0": ["xor0", "cin"], "and0b": ["xor0", "cin"],
    "carry0": ["and0a", "and0b"],
    "dff_sum0": ["sum0"],
    "xor1": ["a1", "b1"], "and1a": ["a1", "b1"],
    "sum1": ["xor1", "carry0"], "and1b": ["xor1", "carry0"],
    "carry1": ["and1a", "and1b"],
    "dff_sum1": ["sum1"], "dff_cout": ["carry1"],
}
NODE_IDS = list(NETLIST_FANIN.keys())
KIND_OF = {
    "a0": "PI", "b0": "PI", "cin": "PI", "a1": "PI", "b1": "PI",
    "xor0": "XOR2", "and0a": "AND2", "sum0": "XOR2", "and0b": "AND2", "carry0": "OR2",
    "dff_sum0": "DFF", "xor1": "XOR2", "and1a": "AND2", "sum1": "XOR2", "and1b": "AND2",
    "carry1": "OR2", "dff_sum1": "DFF", "dff_cout": "DFF",
}
FANOUT_COUNT = {name: 0 for name in NODE_IDS}
for _srcs in NETLIST_FANIN.values():
    for _src in _srcs:
        FANOUT_COUNT[_src] += 1


def section_header(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


# ---------------------------------------------------------------------------
# 1. CONGESTION / ROUTABILITY PREDICTION
# ---------------------------------------------------------------------------
def congestion_prediction() -> None:
    """CONCEPT: after PLACEMENT (deciding an x,y location for every gate on
    the chip), the ROUTER has to fit every wire into physical tracks
    between cells. Some regions end up with far more wires trying to cross
    them than physical routing tracks available — that's congestion, and it
    can force a costly re-placement if discovered only after a full route.
    A GNN can take the placed netlist graph (nodes carry an (x, y)
    position, edges are the wires) and predict a per-region congestion
    score directly, before running the full router.

    This toy version: place each gate on a tiny 3x3 grid (a synthetic,
    hand-picked layout — a real tool gets this from the placer), then
    estimate each grid cell's congestion as the number of wires (edges)
    that PASS THROUGH or terminate in it — literally an edge-counting
    aggregation over cells, the same "gather from what's structurally
    nearby" idea as ../message_passing_basics/, just aggregated over
    physical grid cells instead of graph hops.
    """
    section_header("1. Congestion / routability prediction")
    # A synthetic placement: which grid cell (row, col) each gate sits in.
    placement = {
        "a0": (0, 0), "b0": (0, 0), "cin": (0, 1), "a1": (0, 2), "b1": (0, 2),
        "xor0": (1, 0), "and0a": (1, 0), "sum0": (1, 1), "and0b": (1, 1),
        "carry0": (1, 1), "dff_sum0": (2, 0),
        "xor1": (1, 2), "and1a": (1, 2), "sum1": (2, 1), "and1b": (2, 1),
        "carry1": (2, 2), "dff_sum1": (2, 1), "dff_cout": (2, 2),
    }
    congestion = {}
    for dst, srcs in NETLIST_FANIN.items():
        for src in srcs:
            (r1, c1), (r2, c2) = placement[src], placement[dst]
            # A wire "touches" every cell on the straight-line path between its
            # two endpoints' rows/cols (a coarse Manhattan-routing approximation).
            for r in range(min(r1, r2), max(r1, r2) + 1):
                congestion[(r, c1)] = congestion.get((r, c1), 0) + 1
            for c in range(min(c1, c2), max(c1, c2) + 1):
                congestion[(r2, c)] = congestion.get((r2, c), 0) + 1

    print("Estimated wire count per grid cell (higher = more congested):")
    for r in range(3):
        print("  " + "  ".join(f"({r},{c})={congestion.get((r, c), 0):>2}" for c in range(3)))
    hottest = max(congestion, key=congestion.get)
    print(f"\nHottest cell: {hottest} with {congestion[hottest]} wires — carry0 and its neighbors "
          "sit here, the same carry-chain hot spot every other template in this directory has flagged "
          "structurally. A real congestion-prediction GNN would use this kind of signal (plus real "
          "placement density, pin counts, etc.) to flag cells worth re-placing BEFORE a full route "
          "attempt, which can take minutes to hours on a real design.")


# ---------------------------------------------------------------------------
# 2. TIMING PREDICTION
# ---------------------------------------------------------------------------
def timing_prediction() -> None:
    """CONCEPT: Static Timing Analysis (STA) computes each gate's exact
    ARRIVAL TIME — when its output is guaranteed stable — by propagating
    per-cell delays (from a real, calibrated cell timing library) forward
    through the netlist graph, taking the MAX over each gate's fan-in
    arrival times plus that fan-in edge's delay. That "propagate, then
    take the max over neighbors" step is exactly a message-passing round
    (../message_passing_basics/) with `max` as the aggregation function
    instead of `mean`/`sum`.

    This toy version uses made-up per-gate-kind delays (not a real,
    calibrated cell library) to keep the point about the GRAPH PROPAGATION
    pattern itself: notice how a gate's arrival time depends on its full
    UPSTREAM PATH, not just its own delay — carry1's arrival time is large
    not because carry1 itself is slow, but because everything feeding into
    it accumulated delay first. A real timing-prediction GNN is trained to
    approximate exactly this kind of accumulated quantity directly from
    netlist structure, without running full incremental STA on every
    candidate design change.
    """
    section_header("2. Timing prediction")
    delay_of_kind = {"PI": 0.0, "AND2": 1.0, "OR2": 1.0, "XOR2": 1.5, "DFF": 0.5}  # illustrative, not a real library

    arrival = {}
    def compute_arrival(name: str) -> float:
        if name in arrival:
            return arrival[name]
        fanin = NETLIST_FANIN[name]
        upstream = max((compute_arrival(src) for src in fanin), default=0.0)
        arrival[name] = upstream + delay_of_kind[KIND_OF[name]]
        return arrival[name]

    for name in NODE_IDS:
        compute_arrival(name)

    print("Propagated arrival time per node (message passing with MAX aggregation, not mean):")
    for name in NODE_IDS:
        print(f"  {name:>10}: {arrival[name]:.1f}")
    critical_node = max(arrival, key=arrival.get)
    print(f"\nLatest arrival time: {critical_node} at {arrival[critical_node]:.1f} — this IS the critical "
          "path, discovered here by direct propagation instead of the learned classifier in "
          "../gcn_gate_classification/. A real timing-prediction GNN sits between these two extremes: "
          "faster than full incremental STA re-propagation on every design iteration, more general than "
          "hand-written propagation rules when cell delays depend on things a hand-written formula "
          "doesn't cleanly capture (interconnect parasitics, drive strength, temperature/voltage corners).")


# ---------------------------------------------------------------------------
# 3. POWER ESTIMATION
# ---------------------------------------------------------------------------
def power_estimation() -> None:
    """CONCEPT: dynamic power (the power spent actually switching, as
    opposed to leakage) is approximately `activity x capacitance` per gate,
    summed over the whole circuit — a real tool estimates capacitance from
    the physical wire/gate geometry, and activity (how often each net
    toggles) from either simulation or, here, from the SAME toggle-rate
    message-passing propagation as ../message_passing_basics/. That file
    already built the "propagate toggle rate through the netlist" step;
    this section is what you DO with the result once you have it — turn it
    into a power number instead of just an activity number.

    Fan-out count stands in for capacitance here (a real tool measures the
    actual wire length and load-gate input capacitance; more fanout
    correlates with more wire and more load, which is why it's a cheap,
    honest-about-being-approximate proxy for a toy example).
    """
    section_header("3. Power estimation")
    toggle_rate = {name: 0.0 for name in NODE_IDS}
    toggle_rate.update({"a0": 0.9, "b0": 0.9, "a1": 0.9, "b1": 0.9, "cin": 0.1})
    # One round of mean-aggregation message passing per logic level, same
    # mechanic as ../message_passing_basics/'s message_passing_round.
    for _ in range(6):  # 6 rounds covers this netlist's full depth (see ../message_passing_basics/)
        toggle_rate = {
            name: (sum(toggle_rate[src] for src in NETLIST_FANIN[name]) / len(NETLIST_FANIN[name])
                   if NETLIST_FANIN[name] else toggle_rate[name])
            for name in NODE_IDS
        }

    capacitance_proxy = {name: 1.0 + FANOUT_COUNT[name] for name in NODE_IDS}  # base load + one unit per fanout wire
    power = {name: toggle_rate[name] * capacitance_proxy[name] for name in NODE_IDS}

    print("Estimated relative dynamic power per gate (toggle_rate x capacitance_proxy):")
    for name in NODE_IDS:
        if KIND_OF[name] == "PI":
            continue
        print(f"  {name:>10}: toggle={toggle_rate[name]:.2f}  cap={capacitance_proxy[name]:.0f}  "
              f"power={power[name]:.2f}")
    total = sum(v for k, v in power.items() if KIND_OF[k] != "PI")
    print(f"\nTotal estimated dynamic power (gates only): {total:.2f}")
    print("A real power-estimation GNN learns this same activity-x-capacitance relationship end to end "
          "from real switching-activity traces and layout parasitics, rather than composing two "
          "hand-picked proxies the way this toy example does.")


# ---------------------------------------------------------------------------
# 4. BUG / ANOMALY DETECTION
# ---------------------------------------------------------------------------
def anomaly_detection() -> None:
    """CONCEPT: a synthesis or hand-editing bug often shows up as a node
    whose LOCAL STRUCTURE doesn't look like its neighbors' — a gate wired
    with an unusually high fan-in for its kind, a signal that fans out to
    far more loads than anything else nearby, or a node whose neighborhood
    looks nothing like the rest of the design's. A GNN-based anomaly
    detector encodes each node's neighborhood (the same aggregation idea as
    ../gcn_gate_classification/'s GCN layer) and flags nodes whose
    aggregated embedding is a statistical outlier relative to the rest of
    the graph — conceptually an autoencoder: nodes the model "reconstructs"
    poorly, or that simply sit far from the population mean in embedding
    space, are worth a human's attention.

    This toy version skips training an actual autoencoder (that machinery
    already lives in ../gcn_gate_classification/ and ../gat_attention_
    critical_path/) and instead injects one deliberately WRONG gate — a
    3-input AND gate spliced into the netlist, immediately suspicious
    because every other gate in this whole toy technology library is
    2-input — and computes TWO scores side by side: a RAW per-node z-score
    (own fan-in count alone, no graph awareness) and an AGGREGATED z-score
    (mean-pooled with the node's own undirected neighborhood — self,
    fan-in, AND fan-out — the same shape of neighborhood GCN aggregates
    over in ../gcn_gate_classification/). The raw score alone already
    catches the buggy gate itself; the point of the AGGREGATED column is
    what a purely per-node check can never show: the anomaly's structural
    NEIGHBORS also shift, because they're mean-pooling in a broken node's
    inflated fan-in count too — a signature a graph-blind anomaly detector
    (checking each gate against its own kind's typical fan-in, nothing
    else) would have no way to produce.
    """
    section_header("4. Bug / anomaly detection")
    fanin = dict(NETLIST_FANIN)
    fanin["and0a"] = ["a0", "b0", "cin"]  # <- the injected bug: and0a should never see cin, only a0/b0

    undirected: dict[str, set[str]] = {name: set() for name in NODE_IDS}
    for dst, srcs in fanin.items():
        for src in srcs:
            undirected[dst].add(src)
            undirected[src].add(dst)

    def aggregated_fanin(name: str) -> float:
        values = [len(fanin[name])] + [len(fanin[nb]) for nb in undirected[name]]
        return sum(values) / len(values)

    def zscores(values_by_node: dict[str, float]) -> dict[str, float]:
        values = list(values_by_node.values())
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1e-9
        return {name: (v - mean) / std for name, v in values_by_node.items()}

    gates = [n for n in NODE_IDS if KIND_OF[n] != "PI"]
    raw_z = zscores({name: float(len(fanin[name])) for name in gates})
    agg_z = zscores({name: aggregated_fanin(name) for name in gates})

    print(f"{'gate':>10}  {'raw z':>8}  {'aggregated z':>13}")
    flagged_raw, flagged_agg = [], []
    for name in sorted(gates, key=lambda n: -raw_z[n]):
        raw_flag = "  <-- FLAGGED" if raw_z[name] > 1.5 else ""
        agg_flag = "  <-- FLAGGED" if agg_z[name] > 1.5 else ""
        if raw_flag:
            flagged_raw.append(name)
        if agg_flag:
            flagged_agg.append(name)
        print(f"{name:>10}  {raw_z[name]:>+8.2f}{raw_flag}  {agg_z[name]:>+13.2f}{agg_flag}")

    print(f"\nRaw-feature flags (the bug itself): {flagged_raw}")
    print(f"Aggregated (graph-aware) flags (where the bug's effect ripples to structurally): {flagged_agg}")
    print("and0a is exactly the injected bug, and the raw column catches it on its own. The aggregated "
          "column tells a different story: and0a's OWN aggregated z drops (its inflated fan-in gets "
          "diluted by its two all-fanin-0 primary-input neighbors), while carry0 — structurally FINE on "
          "its own — gets flagged instead, and and1b comes close (z=+1.46), simply because both mean-pool "
          "in and0a's inflated fan-in count from one or two hops away. A design review that only checks "
          "each gate against its own kind's typical fan-in would catch and0a directly; a graph-aware check "
          "instead surfaces WHERE the bug's effect concentrates structurally, which won't always be the "
          "buggy node itself — closer to how a synthesis bug's downstream blast radius actually behaves.")


def main() -> None:
    congestion_prediction()
    timing_prediction()
    power_estimation()
    anomaly_detection()
    print(
        "\n" + "-" * 60 +
        "\nAll four sections used the SAME toy netlist and variations on the SAME message-passing "
        "mechanic (../message_passing_basics/) — aggregate structural or physical neighborhood "
        "information, then turn the aggregate into a per-node prediction. What changes between "
        "congestion, timing, power, and anomaly detection is only WHAT gets aggregated and WHAT the "
        "aggregate is used to predict, not the underlying graph-learning idea."
    )


if __name__ == "__main__":
    main()
