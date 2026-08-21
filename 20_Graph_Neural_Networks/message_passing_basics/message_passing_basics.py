"""
CONCEPT: Message passing — the single mechanic every GNN in this directory
(GCN, GraphSAGE, GAT) is a variation of. One "round" of message passing does
exactly two things for every node at once:

  1. GATHER: collect a message from each of the node's neighbors (here:
     its fan-in gates — the gates whose output wires drive this node).
  2. UPDATE: combine those gathered messages with the node's own current
     feature to produce its NEW feature.

Stack K rounds and a node's final feature has been influenced by every
node within K hops upstream of it — exactly the way a real signal's
switching activity at a flip-flop is a function of every gate in its
fan-in cone, not just its immediate driver. That's the physical intuition
this template makes concrete: we propagate a toggle-rate-like activity
value from the primary inputs (PIs) through the netlist from
../graph_representation_of_netlists/graph_representation_of_netlists.py
(rebuilt inline below, per this repo's convention — see that file's
docstring) and watch it reach further gates with every additional round.

This is deliberately the simplest possible version — no learned weights,
no nonlinearity, just gather-then-average. ../gcn_gate_classification/
adds a learned linear layer on top of this same gather step (that's
literally what a GCN layer is: message passing + a weight matrix + ReLU).
../graphsage_inductive_netlist_embedding/ changes the GATHER step to a
random *sample* of neighbors instead of all of them. ../gat_attention_
critical_path/ changes it again to a *weighted* average, where the weights
are learned per-neighbor instead of fixed at 1/degree.

Run this file directly to watch a signal spread hop by hop, then watch
what happens if you keep going well past the point where it's useful
(over-smoothing):

    python3 message_passing_basics.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Same toy 2-bit ripple-carry adder as ../graph_representation_of_netlists/,
# condensed to just what this file needs: node order + directed fan-in
# edges. See that file for the full annotated version and the ASCII diagram
# of this exact circuit.
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


def message_passing_round(features: dict[str, float]) -> dict[str, float]:
    """One round of gather-then-update, for every node simultaneously.

    GATHER: a node's message is the MEAN of its fan-in neighbors' current
    features (mean, not sum, keeps values from exploding as fan-in grows —
    the same reason GCN normalizes by degree, see ../gcn_gate_classification/).
    A primary input has no fan-in, so it gathers nothing.

    UPDATE: here, simply "replace with the gathered mean" for any node that
    HAS fan-in, and leave primary inputs untouched (a PI's activity is an
    external stimulus, not something computed from the circuit). A real GNN
    layer's update is `combine(self_feature, gathered_message)` through a
    learned weight matrix — see ../gcn_gate_classification/'s `gcn_layer`
    for that next step. This bare-bones version isolates the gather/spread
    behavior on its own, before any learning enters the picture.

    CONCEPT: this is applied to EVERY node in one synchronous round,
    regardless of topological/logic order. A real GNN has no notion of
    "this gate comes before that one" — it just stacks K identical rounds
    and lets the graph structure determine how far information travels.
    """
    new_features = dict(features)
    for node in NODE_IDS:
        fanin = NETLIST_FANIN[node]
        if not fanin:
            continue  # primary input: nothing upstream to gather
        new_features[node] = sum(features[src] for src in fanin) / len(fanin)
    return new_features


# ---------------------------------------------------------------------------
# Undirected neighborhoods (each wire counted both ways) plus a self-loop on
# every node. This is what a real GCN layer actually aggregates over (see
# ../gcn_gate_classification/) — spectral graph convolution is defined on a
# SYMMETRIC adjacency, not the directed one signals actually flow through.
# We use it below purely to demonstrate over-smoothing correctly: the
# directed, fan-in-only `message_passing_round` above reaches a stable fixed
# point once information has fully propagated (round 5) and then stays put
# forever — each node's fixed value differs because a DAG has no cycles to
# average across, so "more rounds" past that point is harmless, not harmful.
# Over-smoothing is a real phenomenon of the UNDIRECTED, self-looped
# aggregation GCN-style layers actually use, shown below instead.
# ---------------------------------------------------------------------------
UNDIRECTED_NEIGHBORS: dict[str, set[str]] = {name: {name} for name in NODE_IDS}  # self-loop included
for _dst, _srcs in NETLIST_FANIN.items():
    for _src in _srcs:
        UNDIRECTED_NEIGHBORS[_dst].add(_src)
        UNDIRECTED_NEIGHBORS[_src].add(_dst)


def message_passing_round_undirected(features: dict[str, float]) -> dict[str, float]:
    """Same gather-then-average idea, but over the UNDIRECTED, self-looped
    neighborhood every node sits in — the shape of aggregation a real GCN
    layer uses (`../gcn_gate_classification/`'s normalized adjacency is
    this same idea, just weighted by degree instead of a plain mean).
    """
    return {node: sum(features[nb] for nb in UNDIRECTED_NEIGHBORS[node]) / len(UNDIRECTED_NEIGHBORS[node])
            for node in NODE_IDS}


def run_rounds(initial: dict[str, float], num_rounds: int, watch: list[str], round_fn=message_passing_round) -> None:
    """Run `num_rounds` of message passing, printing the watched nodes'
    values after each round so you can see the receptive field grow.
    """
    features = dict(initial)
    header = "round".ljust(7) + "".join(name.rjust(12) for name in watch)
    print(header)
    print("-" * len(header))
    print("0".ljust(7) + "".join(f"{features[n]:.3f}".rjust(12) for n in watch))
    for r in range(1, num_rounds + 1):
        features = round_fn(features)
        print(str(r).ljust(7) + "".join(f"{features[n]:.3f}".rjust(12) for n in watch))
    print()


def main() -> None:
    # Toggle rate: fraction of clock cycles each primary input flips value.
    # a0/b0/a1/b1 are "busy" data inputs (toggle often); cin is a steady
    # carry-in that rarely changes. Every gate starts at 0.0 (unknown/no
    # activity observed yet) — message passing is what fills these in from
    # the PI values, exactly like activity propagation in a real power/
    # timing estimation tool (see ../gnn_applications_eda/'s power section).
    initial = {name: 0.0 for name in NODE_IDS}
    initial.update({"a0": 0.9, "b0": 0.9, "a1": 0.9, "b1": 0.9, "cin": 0.1})

    watch = ["cin", "and0b", "carry0", "and1b", "carry1", "dff_cout"]
    # This chain is exactly cin's fan-out cone: cin -> and0b -> carry0 ->
    # and1b -> carry1 -> dff_cout, i.e. the carry chain identified in
    # ../graph_representation_of_netlists/ as the circuit's longest path.

    print("Toggle-rate activity spreading from primary inputs, round by round:\n")
    print(f"Initial PI values: {[(n, initial[n]) for n in ['a0', 'b0', 'cin', 'a1', 'b1']]}\n")
    run_rounds(initial, num_rounds=5, watch=watch)

    print(
        "Notice: cin's low toggle rate (0.1) doesn't show up in carry0 until round 2 (2 hops away:\n"
        "cin -> and0b -> carry0), doesn't reach carry1 until round 4, and dff_cout until round 5.\n"
        "This is the SAME reason a K-layer GNN can only 'see' K hops of graph structure — a node's\n"
        "final embedding is blind to anything further than K hops away, no matter how many features\n"
        "or how much training data you throw at it. Real GNN stacks are usually only 2-4 layers deep\n"
        "for exactly this reason (see 'Limited depth' in this topic's parent README).\n"
    )

    # CONCEPT: over-smoothing. Directed, fan-in-only aggregation (above)
    # reaches a stable fixed point and stops changing — harmless, because a
    # DAG has no cycles to keep re-mixing values through. A real GCN layer
    # (../gcn_gate_classification/) aggregates over the UNDIRECTED,
    # self-looped neighborhood instead (spectral convolution is defined
    # that way), and THAT is where stacking too many rounds becomes
    # actively harmful: repeated undirected averaging is a random walk
    # toward the graph's single global stationary value, and every node's
    # feature converges to it regardless of the node's own role.
    print("Same PI values, now averaged over UNDIRECTED neighborhoods (self-loop included) — the\n"
          "aggregation shape a real GCN layer uses — run for many more rounds:\n")
    run_rounds(initial, num_rounds=30, watch=["and0b", "carry0", "and1b", "carry1"],
               round_fn=message_passing_round_undirected)
    print(
        "By round ~20, and0b/carry0/and1b/carry1 have all converged to nearly the same value (~0.17,\n"
        "the graph-wide stationary average of the PI toggle rates) — this IS real over-smoothing.\n"
        "Any distinguishing structural signal (carry0's higher fan-out, its role on the critical path)\n"
        "has been averaged away. This is why real GNN architectures cap depth at a handful of layers\n"
        "rather than 'more rounds = more accuracy.'"
    )


if __name__ == "__main__":
    main()
