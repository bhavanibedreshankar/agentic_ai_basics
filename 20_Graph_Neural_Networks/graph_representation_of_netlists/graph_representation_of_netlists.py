"""
CONCEPT: Graph representation of an RTL netlist — before any GNN can run over
a circuit, the circuit has to become a graph. A GNN doesn't see Verilog, or
gates, or wires; it sees NODES (each with a feature vector) and EDGES (each
a directed connection between two node ids). This template is entirely about
that translation step: toy netlist -> {nodes, edge_index, node_features}.

Why frame this in RTL at all? A synthesized digital circuit already IS a
graph: every gate/flip-flop is a node, every wire is a directed edge from
the driving gate's output to the driven gate's input. Real EDA (Electronic
Design Automation) tools exploit this directly — GNNs are an active research
area for predicting timing, congestion, and power straight from this graph,
without running the full, slow physical-design flow. See
./gnn_applications_eda.py (../gnn_applications_eda/gnn_applications_eda.py)
for a survey of those use cases; this file only builds the graph they'd
all start from.

Contrast with ../../09_RAG_and_Knowledge/graph_rag/graph_construction.py,
which builds a graph of a SOFTWARE codebase (modules/classes/functions,
contains/calls/inherits/imports edges) by walking a Python AST. The
mechanics are the same shape (nodes dict + edge list, built by walking a
structured description) but the domain graph is completely different: here
a node is a hardware gate driven by concrete boolean logic, and an edge is
a physical wire, not a naming relationship.

The toy circuit used throughout this whole topic directory (all 6 templates
reference it) is a 2-bit ripple-carry adder computing SUM = A + B + CIN:

    a0 b0  cin                 a1 b1
     \\ /  /  \\                  \\ /
     xor0   (also -> and0a)     xor1
      |  \\___________            |  \\
      |         \\      \\         |    \\
    (also        and0b  and0a  (also   and1b
     -> sum0)      \\    /      -> sum1)  \\
                    carry0 ---------------(also -> and1b)
        |                                    |
      sum0 -> DFF_sum0            sum1(=xor1^carry0) -> DFF_sum1
                                  carry1(=and1a|and1b) -> DFF_cout

`carry0` is the interesting node: it is the ONLY signal that crosses from
bit 0's logic into bit 1's logic, so it sits on the circuit's longest
combinational path (the "carry chain") and fans out to two different
gates (sum1, and1b) — more than any other combinational gate in the
circuit, all of them single-fanout. Every later template in this directory
(message passing, GCN,
GraphSAGE, GAT) uses `carry0`'s special structural role as the running
example of "what a GNN can learn to notice."

Run this file directly to build the graph and print a full summary:

    python3 graph_representation_of_netlists.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# CONCEPT: node "type" and node "features". A real netlist node carries both
# a discrete kind (what gate is this?) and continuous attributes useful for
# prediction (fanin count, estimated delay, toggle rate, ...). We keep both:
# `kind` is symbolic/human-readable, `features` is the numeric vector a GNN
# would actually consume. One-hot encoding the gate kind is what makes a
# categorical fact ("this is a XOR2") usable inside numeric linear algebra —
# exactly the same trick embeddings use to turn tokens into vectors.
# ---------------------------------------------------------------------------
GATE_KINDS = ["PI", "AND2", "OR2", "XOR2", "DFF"]  # PI = primary input, DFF = flip-flop (output register)


def _one_hot(kind: str) -> list[float]:
    """Turn a gate kind string into a one-hot vector over GATE_KINDS.
    This IS the node's "type feature" — position `i` is 1.0 iff kind ==
    GATE_KINDS[i]. A real netlist feature vector would append more numeric
    columns (fanin count, logic depth, area, ...) after this one-hot block;
    see `build_feature_matrix` below for exactly that.
    """
    vec = [0.0] * len(GATE_KINDS)
    vec[GATE_KINDS.index(kind)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# The toy netlist itself, described the way a gate-level Verilog netlist
# would define it: each gate lists the *names* of the wires driving its
# inputs. This mirrors how a real netlist parser (e.g. reading a Verilog
# `.v` file's structural instances) would see a design before building a
# graph out of it.
# ---------------------------------------------------------------------------
NETLIST = {
    # Primary inputs have no fan-in of their own.
    "a0": {"kind": "PI", "fanin": []},
    "b0": {"kind": "PI", "fanin": []},
    "cin": {"kind": "PI", "fanin": []},
    "a1": {"kind": "PI", "fanin": []},
    "b1": {"kind": "PI", "fanin": []},
    # Bit 0 logic (a half adder taking cin as a third input).
    "xor0": {"kind": "XOR2", "fanin": ["a0", "b0"]},
    "and0a": {"kind": "AND2", "fanin": ["a0", "b0"]},
    "sum0": {"kind": "XOR2", "fanin": ["xor0", "cin"]},
    "and0b": {"kind": "AND2", "fanin": ["xor0", "cin"]},
    "carry0": {"kind": "OR2", "fanin": ["and0a", "and0b"]},  # <- the carry-chain node
    "dff_sum0": {"kind": "DFF", "fanin": ["sum0"]},
    # Bit 1 logic — carry0 feeds in from bit 0, making it the one signal
    # that crosses the bit boundary and the reason bit 1's logic can't
    # start until bit 0's carry is ready.
    "xor1": {"kind": "XOR2", "fanin": ["a1", "b1"]},
    "and1a": {"kind": "AND2", "fanin": ["a1", "b1"]},
    "sum1": {"kind": "XOR2", "fanin": ["xor1", "carry0"]},
    "and1b": {"kind": "AND2", "fanin": ["xor1", "carry0"]},
    "carry1": {"kind": "OR2", "fanin": ["and1a", "and1b"]},
    "dff_sum1": {"kind": "DFF", "fanin": ["sum1"]},
    "dff_cout": {"kind": "DFF", "fanin": ["carry1"]},
}

GRAPH_FILE = Path(__file__).parent / "netlist_graph.json"


def build_netlist_graph(netlist: dict = NETLIST) -> dict:
    """Convert the fanin-list netlist description above into the standard
    graph form every GNN framework expects (this is deliberately shaped
    like a PyTorch Geometric `Data` object, without requiring the
    `torch_geometric` dependency — see the docstring note at the bottom):

      - node_ids:  list[str]              stable ordering, index == node id
      - edge_index: list[(int, int)]      directed (src, dst) pairs, wire direction
      - node_kind: list[str]              symbolic gate kind, one per node (for printing)

    CONCEPT: edge DIRECTION matters here in a way it might not in a social
    graph. A wire has a driver and a receiver; the edge always points
    driver -> receiver, i.e. (fanin_gate, this_gate). That directionality is
    exactly what makes message passing in ../message_passing_basics/
    meaningful: information should flow the way signals actually propagate,
    from primary inputs toward flip-flops, not the other way.
    """
    node_ids = list(netlist.keys())
    index_of = {name: i for i, name in enumerate(node_ids)}

    edge_index: list[tuple[int, int]] = []
    for dst_name, info in netlist.items():
        for src_name in info["fanin"]:
            edge_index.append((index_of[src_name], index_of[dst_name]))

    node_kind = [netlist[name]["kind"] for name in node_ids]
    return {"node_ids": node_ids, "index_of": index_of, "edge_index": edge_index, "node_kind": node_kind}


def build_feature_matrix(graph: dict, netlist: dict = NETLIST) -> list[list[float]]:
    """Build the numeric X matrix (one feature row per node) a GNN layer
    actually multiplies against. Columns, in order:

      [one-hot gate kind (5 cols)] + [fanin_count] + [fanout_count]

    Structural counts like fanin/fanout are cheap, always-available
    features for a real netlist GNN (no simulation needed to compute them),
    and they're exactly the kind of signal ../gcn_gate_classification/
    trains a classifier on.
    """
    fanout_count = {name: 0 for name in netlist}
    for info in netlist.values():
        for src_name in info["fanin"]:
            fanout_count[src_name] += 1

    features = []
    for name in graph["node_ids"]:
        info = netlist[name]
        row = _one_hot(info["kind"])
        row.append(float(len(info["fanin"])))     # fanin_count
        row.append(float(fanout_count[name]))       # fanout_count
        features.append(row)
    return features


def describe_graph(graph: dict, features: list[list[float]]) -> None:
    node_ids, edge_index, node_kind = graph["node_ids"], graph["edge_index"], graph["node_kind"]

    by_kind: dict[str, int] = {}
    for kind in node_kind:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"{len(node_ids)} nodes: " + ", ".join(f"{count} {kind}" for kind, count in by_kind.items()))
    print(f"{len(edge_index)} directed edges (wires)\n")

    print("Wires (driver --> receiver):")
    for src_i, dst_i in edge_index:
        print(f"  {node_ids[src_i]:>10} --> {node_ids[dst_i]}")

    print("\nFeature vector layout: [one-hot(PI,AND2,OR2,XOR2,DFF)] + [fanin_count] + [fanout_count]")
    for name, row in zip(node_ids, features):
        print(f"  {name:>10}: {row}")

    # The payoff: carry0 is structurally distinctive with zero simulation —
    # its fanout_count (3) is the highest of any combinational gate, purely
    # from graph shape. That's the same kind of structural clue a GCN in
    # ../gcn_gate_classification/gcn_gate_classification.py learns to weigh.
    carry0_row = features[graph["index_of"]["carry0"]]
    print(f"\ncarry0's fanout_count = {int(carry0_row[-1])} (tied for highest, but the ONLY gate "
          "whose fanout crosses from bit 0's logic into bit 1's) — it drives both sum1 and and1b, "
          "which is exactly why it sits on the circuit's longest (carry-chain) path.")


def main() -> None:
    graph = build_netlist_graph()
    features = build_feature_matrix(graph)
    print("Built a graph from a toy 2-bit ripple-carry adder netlist.\n")
    describe_graph(graph, features)

    payload = {
        "node_ids": graph["node_ids"],
        "edge_index": graph["edge_index"],
        "node_kind": graph["node_kind"],
        "features": features,
    }
    GRAPH_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {GRAPH_FILE.name} — every other template in this directory rebuilds an "
          "equivalent graph inline (this repo's convention, see ../../09_RAG_and_Knowledge/graph_rag/ "
          "for the same pattern) rather than importing this file, so each stays runnable standalone.")

    print(
        "\nNote on PyTorch Geometric: a real project would likely represent this same graph as a "
        "torch_geometric.data.Data(x=<features tensor>, edge_index=<[2, num_edges] tensor>) and run "
        "it through torch_geometric.nn layers (GCNConv, SAGEConv, GATConv). This template (and the "
        "rest of this directory) reimplements the same math in plain Python lists/loops instead of "
        "requiring the torch_geometric dependency — the node_ids/edge_index/features shapes above map "
        "directly onto that Data object's fields if you later port this to real PyG code."
    )


if __name__ == "__main__":
    main()
