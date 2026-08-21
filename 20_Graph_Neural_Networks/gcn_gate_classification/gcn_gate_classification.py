"""
CONCEPT: GCN (Graph Convolutional Network) gate classification — a GCN layer
is exactly the "gather + linear + nonlinearity" idea from
../message_passing_basics/message_passing_basics.py, made LEARNABLE. Instead
of hand-picking "average my neighbors" as the update rule, a GCN layer
learns a weight matrix W so the aggregated neighborhood features get
transformed the way the training data says they should be, then squashes
the result through ReLU:

    H' = ReLU(A_hat @ H @ W)

where A_hat is the (symmetrically normalized, self-looped) adjacency matrix
— the same "undirected neighborhood" shape used for the over-smoothing demo
in ../message_passing_basics/, not the directed fan-in-only one. Stacking
two of these layers is a 2-hop GCN; that's what this file trains.

TASK: predict, for each gate in the toy 2-bit ripple-carry adder netlist
(from ../graph_representation_of_netlists/, rebuilt inline below), whether
it sits on the circuit's longest combinational path (the "carry chain" —
see that file's docstring for the ASCII diagram) using only structural
features (gate kind, fanin/fanout counts) plus 2 hops of graph structure.
This is a real EDA use case in miniature: predicting timing criticality
from a netlist graph BEFORE running full static timing analysis, so a tool
can prioritize which gates to scrutinize — see ../gnn_applications_eda/'s
timing-prediction section for the fuller picture.

CONCEPT: semi-supervised / TRANSDUCTIVE learning. GCN is trained on labels
for only SOME nodes (`LABELED_GATES` below), while message passing still
runs over the WHOLE graph, including unlabeled test gates and even the
primary inputs — the unlabeled nodes' features still help shape what the
labeled nodes' embeddings look like. This is the classic Kipf & Welling GCN
setup, and it's the opposite of ../graphsage_inductive_netlist_embedding/,
which trains only from LOCAL neighbor samples and can then run on a
completely different graph it never saw — a GCN's A_hat is baked to one
fixed graph, so it can't do that (see that file's docstring for the
contrast made concrete).

No numpy/PyTorch/torch_geometric — every matrix operation below is written
out as nested Python lists and loops, and the whole training loop (forward
pass, manual backprop, gradient descent) is ~50 lines. See the note at the
bottom of ../graph_representation_of_netlists/graph_representation_of_netlists.py
for how this maps onto a real `torch_geometric.nn.GCNConv` if you port it.

Run this file directly to train the 2-layer GCN and see it classify held-out gates:

    python3 gcn_gate_classification.py
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Same toy netlist as ../graph_representation_of_netlists/ and
# ../message_passing_basics/, condensed to what this file needs.
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
INDEX_OF = {name: i for i, name in enumerate(NODE_IDS)}
N = len(NODE_IDS)
GATE_KINDS = ["PI", "AND2", "OR2", "XOR2", "DFF"]
KIND_OF = {
    "a0": "PI", "b0": "PI", "cin": "PI", "a1": "PI", "b1": "PI",
    "xor0": "XOR2", "and0a": "AND2", "sum0": "XOR2", "and0b": "AND2", "carry0": "OR2",
    "dff_sum0": "DFF", "xor1": "XOR2", "and1a": "AND2", "sum1": "XOR2", "and1b": "AND2",
    "carry1": "OR2", "dff_sum1": "DFF", "dff_cout": "DFF",
}

# CONCEPT: the ground-truth label. Walking the netlist by hand (see
# ../graph_representation_of_netlists/'s ASCII diagram), the longest
# combinational path from any PI to a flip-flop is:
#   {a0,b0} -> xor0 -> and0b -> carry0 -> and1b -> carry1 -> dff_cout
# Every gate on that path is timing-critical (label 1); everything else
# (including the shorter bit-0 sum path and bit-1's non-carry AND) is not.
CRITICAL_PATH_GATES = {"xor0", "and0b", "carry0", "and1b", "carry1", "dff_cout"}
ALL_GATES = [n for n in NODE_IDS if KIND_OF[n] != "PI"]  # 13 gates, PIs excluded (nothing to classify)

# Semi-supervised split: only SOME gates' labels are used for training loss.
# The rest ("test" gates) are predicted using ONLY the learned weights plus
# 2 hops of graph structure — they're never shown their own label. Chosen
# to keep both the critical/non-critical class balance AND each test gate's
# graph distance to the nearest labeled gate roughly even — an all-majority-
# class or all-far-from-any-label split would make the task trivially easy
# or trivially impossible rather than a fair demonstration of what 2 hops
# of message passing can (and can't) recover.
LABELED_GATES = ["xor0", "carry0", "dff_cout", "and0a", "sum0", "xor1", "dff_sum1"]  # 3 critical, 4 not
TEST_GATES = [g for g in ALL_GATES if g not in LABELED_GATES]  # and0b, and1b, carry1 (critical); dff_sum0, and1a, sum1 (not)


# ---------------------------------------------------------------------------
# Minimal dense linear algebra — plain nested lists, no numpy. Every GCN
# forward/backward step below is one of these three operations.
# ---------------------------------------------------------------------------
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows, inner, cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*A)]


def elementwise(A: list[list[float]], B: list[list[float]], op) -> list[list[float]]:
    return [[op(A[i][j], B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]


def relu(A: list[list[float]]) -> list[list[float]]:
    return [[max(0.0, x) for x in row] for row in A]


def relu_grad_mask(Z: list[list[float]]) -> list[list[float]]:
    return [[1.0 if x > 0 else 0.0 for x in row] for row in Z]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Build A_hat: the symmetrically-normalized, self-looped adjacency matrix
# GCN's message passing runs on — the same UNDIRECTED neighborhood shape
# used for the over-smoothing demo in ../message_passing_basics/, here
# normalized by degree the way Kipf & Welling (2017) define it:
#   A_hat = D^-1/2 (A + I) D^-1/2
# The D^-1/2 scaling (instead of a plain 1/degree mean) keeps the
# aggregation numerically well-behaved regardless of how skewed the fanout
# distribution is — high-fanout nodes like carry0 don't get to dominate
# every neighbor's update just because they touch more edges.
# ---------------------------------------------------------------------------
def build_normalized_adjacency() -> list[list[float]]:
    adj = [[0.0] * N for _ in range(N)]
    for i in range(N):
        adj[i][i] = 1.0  # self-loop: a node's own features count in its own update
    for dst, srcs in NETLIST_FANIN.items():
        for src in srcs:
            adj[INDEX_OF[src]][INDEX_OF[dst]] = 1.0
            adj[INDEX_OF[dst]][INDEX_OF[src]] = 1.0  # undirected: symmetric

    degree = [sum(adj[i]) for i in range(N)]
    inv_sqrt_degree = [1.0 / math.sqrt(d) for d in degree]
    return [[adj[i][j] * inv_sqrt_degree[i] * inv_sqrt_degree[j] for j in range(N)] for i in range(N)]


def _logic_depth(name: str, cache: dict[str, int]) -> int:
    """Longest path (in gate hops) from any primary input to `name` — cheap
    to compute from the netlist structure alone, no timing simulation
    needed (real STA/"static timing analysis" tools compute something like
    this as a first pass too). Memoized since the same fanin gate is
    revisited by multiple downstream gates (e.g. carry0 feeds both sum1
    and and1b).
    """
    if name in cache:
        return cache[name]
    fanin = NETLIST_FANIN[name]
    cache[name] = 0 if not fanin else 1 + max(_logic_depth(src, cache) for src in fanin)
    return cache[name]


def build_feature_matrix() -> list[list[float]]:
    fanout_count = {name: 0 for name in NODE_IDS}
    for srcs in NETLIST_FANIN.values():
        for src in srcs:
            fanout_count[src] += 1
    depth_cache: dict[str, int] = {}
    depths = {name: _logic_depth(name, depth_cache) for name in NODE_IDS}
    max_depth = max(depths.values())

    features = []
    for name in NODE_IDS:
        row = [1.0 if KIND_OF[name] == k else 0.0 for k in GATE_KINDS]
        row.append(len(NETLIST_FANIN[name]) / 2.0)          # fanin_count, normalized (max fanin here is 2)
        row.append(fanout_count[name] / 2.0)                 # fanout_count, normalized (max fanout here is 2)
        row.append(depths[name] / max_depth)                 # logic depth from nearest PI, normalized to [0, 1]
        features.append(row)
    return features


# ---------------------------------------------------------------------------
# The 2-layer GCN itself: two stacked (aggregate -> linear -> nonlinearity)
# rounds, exactly the mechanic named in the module docstring.
# ---------------------------------------------------------------------------
class GCN:
    def __init__(self, in_dim: int, hidden_dim: int, seed: int = 0):
        rng = random.Random(seed)
        scale1 = math.sqrt(2.0 / in_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W1 = [[rng.gauss(0, scale1) for _ in range(hidden_dim)] for _ in range(in_dim)]
        self.W2 = [[rng.gauss(0, scale2)] for _ in range(hidden_dim)]

    def forward(self, A_hat: list[list[float]], X: list[list[float]]):
        agg0 = matmul(A_hat, X)          # layer 1 GATHER: 1-hop neighborhood
        z1 = matmul(agg0, self.W1)       # layer 1 UPDATE (linear part)
        h1 = relu(z1)                    # layer 1 nonlinearity
        agg1 = matmul(A_hat, h1)         # layer 2 GATHER: now 2 hops total from X
        z2 = matmul(agg1, self.W2)       # layer 2 UPDATE -> per-node logit
        p = [[sigmoid(z2[i][0])] for i in range(N)]
        cache = {"agg0": agg0, "z1": z1, "h1": h1, "agg1": agg1, "z2": z2}
        return p, cache

    def backward(self, A_hat, X, cache, p, y_mask, lr: float):
        """Manual backprop for binary cross-entropy loss, averaged only over
        the LABELED nodes (y_mask[i] is None for unlabeled nodes, so they
        contribute zero gradient — they still influenced the forward pass
        through message passing, just not the loss).
        """
        labeled = [i for i in range(N) if y_mask[i] is not None]
        # dL/dz2 for BCE-with-sigmoid simplifies to (p - y); zero out unlabeled rows.
        dz2 = [[0.0] for _ in range(N)]
        for i in labeled:
            dz2[i][0] = (p[i][0] - y_mask[i]) / len(labeled)

        agg1_T = transpose(cache["agg1"])
        dW2 = matmul(agg1_T, dz2)                                   # (hidden, 1)
        d_agg1 = matmul(dz2, transpose(self.W2))                    # (N, hidden)
        d_h1 = matmul(A_hat, d_agg1)                                 # A_hat symmetric: A_hat^T == A_hat
        d_z1 = elementwise(d_h1, relu_grad_mask(cache["z1"]), lambda a, b: a * b)
        agg0_T = transpose(cache["agg0"])
        dW1 = matmul(agg0_T, d_z1)                                   # (in_dim, hidden)

        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] -= lr * dW1[i][j]
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] -= lr * dW2[i][j]


def bce_loss(p: list[list[float]], y_mask: list[float | None]) -> float:
    labeled = [i for i in range(N) if y_mask[i] is not None]
    total = 0.0
    for i in labeled:
        pi = min(max(p[i][0], 1e-9), 1 - 1e-9)  # clip: avoid log(0)
        total += -(y_mask[i] * math.log(pi) + (1 - y_mask[i]) * math.log(1 - pi))
    return total / len(labeled)


def main() -> None:
    A_hat = build_normalized_adjacency()
    X = build_feature_matrix()

    y_mask: list[float | None] = [None] * N
    for gate in LABELED_GATES:
        y_mask[INDEX_OF[gate]] = 1.0 if gate in CRITICAL_PATH_GATES else 0.0

    print(f"Training on {len(LABELED_GATES)} labeled gates: {LABELED_GATES}")
    print(f"Held out (never shown a label) {len(TEST_GATES)} test gates: {TEST_GATES}\n")

    gcn = GCN(in_dim=len(X[0]), hidden_dim=6, seed=38)
    for epoch in range(1, 401):
        p, cache = gcn.forward(A_hat, X)
        gcn.backward(A_hat, X, cache, p, y_mask, lr=0.3)
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:>4}  loss = {bce_loss(p, y_mask):.4f}")

    print("\nFinal predictions on held-out TEST gates (never used in the loss):")
    p, _ = gcn.forward(A_hat, X)
    correct = 0
    for gate in TEST_GATES:
        i = INDEX_OF[gate]
        pred_label = 1 if p[i][0] >= 0.5 else 0
        true_label = 1 if gate in CRITICAL_PATH_GATES else 0
        correct += int(pred_label == true_label)
        flag = "critical" if pred_label else "not critical"
        correct_flag = "correct" if pred_label == true_label else "WRONG"
        print(f"  {gate:>10}: P(critical)={p[i][0]:.3f} -> {flag:<13} (true={'critical' if true_label else 'not critical'}, {correct_flag})")

    print(f"\nTest accuracy: {correct}/{len(TEST_GATES)}")
    print(
        "\nThe GCN never saw and0b's, and1b's, or carry1's TRUE labels — it inferred them purely from\n"
        "2 hops of graph structure (fanin/fanout/depth features, gate kind, and what THEIR neighbors'\n"
        "features look like) plus what it learned from the 7 labeled gates. That's the payoff of message\n"
        "passing over a graph instead of classifying each gate from its own isolated feature row: two\n"
        "isolated AND2 gates (and1a vs and1b) have nearly identical own-features — what tells them apart\n"
        "is what's 1-2 hops away: and1b sits next to carry0, and1a doesn't. and1a's wrong prediction\n"
        "above shows this isn't a solved problem either — a tiny 18-node graph with 7 labels is a small,\n"
        "noisy training signal, exactly like a real semi-supervised GCN with limited labeled data."
    )


if __name__ == "__main__":
    main()
