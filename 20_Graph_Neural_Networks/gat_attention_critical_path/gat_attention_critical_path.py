"""
CONCEPT: GAT (Graph Attention Network) — instead of averaging a node's
neighbors with a fixed weight (mean, in ../message_passing_basics/;
degree-normalized, in ../gcn_gate_classification/'s GCN; a random sample,
in ../graphsage_inductive_netlist_embedding/'s GraphSAGE), a GAT layer
LEARNS how much attention each neighbor deserves, per node, per forward
pass:

    e_vu = LeakyReLU(a . [W h_v || W h_u])      -- how relevant is u to v?
    alpha_vu = softmax_u(e_vu)                   -- normalize over v's neighbors
    h_v' = ReLU(sum_u alpha_vu * W h_u)          -- weighted aggregate

That per-neighbor weight `alpha_vu` is exactly the thing this template makes
visible: for a gate with two fan-in signals, does the GAT layer end up
paying more attention to the one that actually matters for the task, or
does it split attention evenly? TASK here is the same critical-path
classification as ../gcn_gate_classification/, on the same toy 2-bit
ripple-carry adder from ../graph_representation_of_netlists/ (rebuilt
inline below).

The headline example is `carry1`, whose two fan-in signals are `and1a` and
`and1b`. Structurally they look almost identical (both AND2 gates, same
fanin/fanout counts) — the ONLY thing that tells them apart is that
`and1b`'s own fan-in includes `carry0`, the carry-chain node, while
`and1a`'s doesn't. A GAT layer that's learned the critical-path task well
should end up assigning `and1b` noticeably higher attention than `and1a`
when computing `carry1`'s embedding — exactly the kind of "which fan-in
signal actually matters for timing" judgment a real static timing
analysis tool has to make explicitly, and which this template's GAT layer
discovers purely from training signal.

Unlike ../gcn_gate_classification/ and ../graphsage_inductive_netlist_
embedding/, attention here is computed over each node's DIRECTED fan-in
only (not the undirected neighborhood) — attention over "which of my
INPUTS matters" is the natural framing for a gate, and it's what makes the
carry1/and1a/and1b story legible: we're asking "how much does each input
matter to this gate's output," not "how much does this gate matter to its
surroundings."

No numpy/PyTorch/PyTorch-Geometric — see the note at the bottom of
../graph_representation_of_netlists/graph_representation_of_netlists.py
for how this maps onto `torch_geometric.nn.GATConv`. Because attention
weights differ per (v, u) pair rather than following one shared matrix
like GCN's A_hat, this file's forward/backward is written as explicit
per-node loops instead of the whole-graph matrix multiplies used in the
other two trainable templates in this directory — still exact, manually
verified analytic gradients (see the README's note on the finite-difference
check used during development), just shaped differently.

Run this file directly to train the 2-layer GAT and inspect carry1's
learned attention weights:

    python3 gat_attention_critical_path.py
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Same toy netlist as every other template in this directory.
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
CRITICAL_PATH_GATES = {"xor0", "and0b", "carry0", "and1b", "carry1", "dff_cout"}

# Attention is computed over each node's fan-in PLUS itself (a self-loop —
# standard in GAT, exactly as in ../gcn_gate_classification/'s A_hat — so a
# primary input, with no fan-in, still attends to (only) itself).
NEIGHBOR_SETS: dict[str, list[str]] = {v: [v] + NETLIST_FANIN[v] for v in NODE_IDS}


def build_feature_matrix() -> list[list[float]]:
    fanout_count = {name: 0 for name in NODE_IDS}
    for srcs in NETLIST_FANIN.values():
        for src in srcs:
            fanout_count[src] += 1
    features = []
    for name in NODE_IDS:
        row = [1.0 if KIND_OF[name] == k else 0.0 for k in GATE_KINDS]
        row.append(len(NETLIST_FANIN[name]) / 2.0)
        row.append(fanout_count[name] / 2.0)
        features.append(row)
    return features


# ---------------------------------------------------------------------------
# Small vector helpers (plain lists, no numpy).
# ---------------------------------------------------------------------------
def dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def vec_add(u: list[float], v: list[float]) -> list[float]:
    return [a + b for a, b in zip(u, v)]


def vec_scale(u: list[float], s: float) -> list[float]:
    return [a * s for a in u]


def leaky_relu(x: float, slope: float = 0.2) -> float:
    return x if x > 0 else slope * x


def leaky_relu_grad(x: float, slope: float = 0.2) -> float:
    return 1.0 if x > 0 else slope


def relu_vec(u: list[float]) -> list[float]:
    return [max(0.0, x) for x in u]


def relu_grad_vec(u: list[float]) -> list[float]:
    return [1.0 if x > 0 else 0.0 for x in u]


def matmul(A, B):
    rows, inner, cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# One GAT layer: linear projection (Wh = X @ W) + per-node attention over
# NEIGHBOR_SETS + weighted aggregate + ReLU. Forward AND backward both
# operate node-by-node since each node's neighbor count/identity differs —
# see the module docstring for why this can't be one whole-graph matmul the
# way GCN's/SAGE's aggregation is.
# ---------------------------------------------------------------------------
class GATLayer:
    def __init__(self, in_dim: int, hidden_dim: int, rng: random.Random):
        scale = math.sqrt(2.0 / in_dim)
        self.W = [[rng.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(in_dim)]
        self.a_src = [rng.gauss(0, 0.5) for _ in range(hidden_dim)]
        self.a_dst = [rng.gauss(0, 0.5) for _ in range(hidden_dim)]
        self.hidden_dim = hidden_dim

    def forward(self, X: list[list[float]]):
        Wh = matmul(X, self.W)  # (N, hidden) — the shared linear projection every attention score is built from
        out = [None] * N
        cache_per_node = [None] * N
        for v_name in NODE_IDS:
            v = INDEX_OF[v_name]
            neighbors = [INDEX_OF[u_name] for u_name in NEIGHBOR_SETS[v_name]]
            pre = [dot(Wh[v], self.a_src) + dot(Wh[u], self.a_dst) for u in neighbors]
            scores = [leaky_relu(p) for p in pre]
            max_score = max(scores)  # numerically-stable softmax
            exp_scores = [math.exp(s - max_score) for s in scores]
            Z = sum(exp_scores)
            alpha = [e / Z for e in exp_scores]
            agg = [0.0] * self.hidden_dim
            for a_u, u in zip(alpha, neighbors):
                agg = vec_add(agg, vec_scale(Wh[u], a_u))
            out[v] = relu_vec(agg)
            cache_per_node[v] = {"neighbors": neighbors, "pre": pre, "alpha": alpha, "agg": agg}
        return out, {"X": X, "Wh": Wh, "per_node": cache_per_node}

    def backward(self, cache, d_out: list[list[float]], lr: float):
        """d_out[v] is dLoss/d(out[v]) from whatever consumed this layer's
        output. Accumulates dWh (per node, from every role a node plays:
        as a v attending outward, AND as a u being attended to by others),
        then converts dWh -> dW via the same X^T @ dWh pattern used in
        ../gcn_gate_classification/ and ../graphsage_inductive_netlist_
        embedding/, plus dX to pass to an earlier layer if stacked.
        """
        Wh = cache["Wh"]
        d_Wh = [[0.0] * self.hidden_dim for _ in range(N)]
        d_a_src = [0.0] * self.hidden_dim
        d_a_dst = [0.0] * self.hidden_dim

        for v_name in NODE_IDS:
            v = INDEX_OF[v_name]
            info = cache["per_node"][v]
            neighbors, alpha, pre, agg = info["neighbors"], info["alpha"], info["pre"], info["agg"]

            d_agg = [d_out[v][k] * (1.0 if agg[k] > 0 else 0.0) for k in range(self.hidden_dim)]  # ReLU grad
            d_alpha = [dot(d_agg, Wh[u]) for u in neighbors]
            for a_u, u in zip(alpha, neighbors):
                d_Wh[u] = vec_add(d_Wh[u], vec_scale(d_agg, a_u))  # weighted-sum term: dWh_u += alpha_u * d_agg

            # Softmax Jacobian-vector product: dScore_u = alpha_u * (dAlpha_u - sum_u' alpha_u' * dAlpha_u')
            weighted_sum = sum(a * da for a, da in zip(alpha, d_alpha))
            d_scores = [a * (da - weighted_sum) for a, da in zip(alpha, d_alpha)]
            d_pre = [ds * leaky_relu_grad(p) for ds, p in zip(d_scores, pre)]

            for dp, u in zip(d_pre, neighbors):
                d_Wh[v] = vec_add(d_Wh[v], vec_scale(self.a_src, dp))  # pre_u = Wh_v . a_src + ...
                d_a_src = vec_add(d_a_src, vec_scale(Wh[v], dp))
                d_Wh[u] = vec_add(d_Wh[u], vec_scale(self.a_dst, dp))  # ... + Wh_u . a_dst
                d_a_dst = vec_add(d_a_dst, vec_scale(Wh[u], dp))

        dW = matmul(transpose(cache["X"]), d_Wh)
        d_X = matmul(d_Wh, transpose(self.W))

        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= lr * dW[i][j]
        for k in range(self.hidden_dim):
            self.a_src[k] -= lr * d_a_src[k]
            self.a_dst[k] -= lr * d_a_dst[k]
        return d_X


def bce_loss(p: list[float], y_mask: list[float | None]) -> float:
    labeled = [i for i in range(N) if y_mask[i] is not None]
    total = 0.0
    for i in labeled:
        pi = min(max(p[i], 1e-9), 1 - 1e-9)
        total += -(y_mask[i] * math.log(pi) + (1 - y_mask[i]) * math.log(1 - pi))
    return total / len(labeled)


def main() -> None:
    rng = random.Random(11)
    X = build_feature_matrix()

    layer1 = GATLayer(in_dim=len(X[0]), hidden_dim=6, rng=rng)
    layer2 = GATLayer(in_dim=6, hidden_dim=6, rng=rng)
    scale_c = math.sqrt(2.0 / 6)
    Wc = [[rng.gauss(0, scale_c)] for _ in range(6)]  # classifier head: 2-hop embedding -> logit

    labeled_gates = ["xor0", "carry0", "dff_cout", "and0a", "sum0", "xor1", "dff_sum1"]
    y_mask: list[float | None] = [None] * N
    for gate in labeled_gates:
        y_mask[INDEX_OF[gate]] = 1.0 if gate in CRITICAL_PATH_GATES else 0.0
    test_gates = [g for g in NODE_IDS if KIND_OF[g] != "PI" and g not in labeled_gates]

    def forward_all():
        h1, cache1 = layer1.forward(X)
        h2, cache2 = layer2.forward(h1)
        logits = matmul(h2, Wc)
        p = [sigmoid(logits[i][0]) for i in range(N)]
        return p, h1, h2, cache1, cache2

    print(f"Training 2-layer GAT on {len(labeled_gates)} labeled gates...")
    for epoch in range(1, 401):
        p, h1, h2, cache1, cache2 = forward_all()

        labeled = [i for i in range(N) if y_mask[i] is not None]
        d_logits = [[0.0] for _ in range(N)]
        for i in labeled:
            d_logits[i][0] = (p[i] - y_mask[i]) / len(labeled)
        dWc = matmul(transpose(h2), d_logits)
        d_h2 = matmul(d_logits, transpose(Wc))

        d_h1 = layer2.backward(cache2, d_h2, lr=0.3)
        layer1.backward(cache1, d_h1, lr=0.3)
        for i in range(len(Wc)):
            Wc[i][0] -= 0.3 * dWc[i][0]

        if epoch % 100 == 0:
            print(f"  epoch {epoch:>4}  loss = {bce_loss(p, y_mask):.4f}")

    print("\nHeld-out test gate predictions:")
    p, h1, h2, cache1, cache2 = forward_all()
    correct = 0
    for gate in test_gates:
        i = INDEX_OF[gate]
        pred = 1 if p[i] >= 0.5 else 0
        true = 1 if gate in CRITICAL_PATH_GATES else 0
        correct += int(pred == true)
        flag, ok = ("critical" if pred else "not critical"), ("correct" if pred == true else "WRONG")
        print(f"  {gate:>10}: P(critical)={p[i]:.3f} -> {flag:<13} (true={'critical' if true else 'not critical'}, {ok})")
    print(f"Test accuracy: {correct}/{len(test_gates)}")

    # --- The headline result: carry1's learned attention over its two fan-in gates ---
    # CONCEPT: why LAYER 2, not layer 1. and1a and and1b have IDENTICAL raw
    # features in X (both AND2, same fanin/fanout counts) — layer 1's
    # attention (over raw X) has nothing to distinguish them with yet, and
    # comes out ~equal by construction (see the README for the actual
    # layer-1 numbers). It's only AFTER layer 1 that and1b's embedding
    # (h1) has absorbed carry0's signal through ITS OWN fan-in aggregation
    # — and1a's h1 hasn't, because a1/b1 aren't on the carry chain. Layer
    # 2's attention over carry1's neighbors operates on THOSE h1 embeddings,
    # which is where the distinguishing signal actually becomes visible.
    print("\ncarry1's fan-in attention weights, LAYER 2 (attention over h1 embeddings, i.e. each\n"
          "neighbor's OWN 1-hop context, not raw features — see the CONCEPT comment above main() call):")
    carry1_info = cache2["per_node"][INDEX_OF["carry1"]]
    for u_idx, alpha in zip(carry1_info["neighbors"], carry1_info["alpha"]):
        name = NODE_IDS[u_idx]
        note = "  <- self" if name == "carry1" else ("  <- feeds from carry0 (critical)" if name == "and1b" else "  <- feeds from a1/b1 only")
        print(f"  alpha(carry1 <- {name:>7}) = {alpha:.3f}{note}")
    print(
        "\nIf alpha(carry1 <- and1b) came out higher than alpha(carry1 <- and1a) above, the attention\n"
        "mechanism singled out the fan-in that actually carries the carry-chain signal — the same\n"
        "judgment a timing engineer would make by hand, discovered here purely from the critical-path\n"
        "training labels, with no hand-coded rule saying 'prefer inputs connected to carry0.'"
    )


if __name__ == "__main__":
    main()
