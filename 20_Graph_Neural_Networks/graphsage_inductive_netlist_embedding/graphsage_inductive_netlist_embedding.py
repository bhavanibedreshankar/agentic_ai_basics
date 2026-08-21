"""
CONCEPT: GraphSAGE ("SAmple and aggreGatE") — inductive node embeddings.
../gcn_gate_classification/'s GCN bakes ONE fixed, global, degree-normalized
adjacency matrix (A_hat) into every forward pass. That matrix's size IS the
number of nodes in the training graph, so a trained GCN literally cannot be
applied to a graph with different nodes — the matrix shapes don't even
match. GraphSAGE fixes this by learning a LOCAL aggregation function
instead of a global matrix:

    h_v' = ReLU(W . concat(h_v, mean(h_u for u in sample(neighbors(v), k))))

That function only ever looks at a node's OWN features and a (possibly
sampled) subset of its immediate neighbors' features — nothing about the
learned weight matrix W depends on how many nodes are in the graph, or
which specific nodes they are. So the exact same trained W can be re-run,
unchanged, on a completely different graph. That's what "inductive" means
here, and this file proves it concretely: train on the toy 2-bit
ripple-carry adder from ../graph_representation_of_netlists/ (rebuilt
inline below), then apply the SAME weights, with ZERO retraining, to a
brand-new 3-bit ripple-carry adder netlist this code has never seen during
training — a netlist with different node names, different node count, and
one node (carry1) with a bigger neighborhood than anything in the training
graph.

Contrast directly with ../gcn_gate_classification/gcn_gate_classification.py:
that file's `A_hat` is transductive — fixed to one graph, computed once,
reused every forward pass. This file's `sample_neighbors` is inductive —
computed fresh for whatever graph you hand it, using the same rule
regardless of graph identity or size.

No numpy/PyTorch/PyTorch-Geometric — same plain-Python matrix helpers as
../gcn_gate_classification/, reused verbatim so the two files' training
loops are easy to compare side by side.

Run this file directly to train on the 2-bit adder, then evaluate
(zero-shot, no retraining) on an unseen 3-bit adder:

    python3 graphsage_inductive_netlist_embedding.py
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Minimal dense linear algebra — identical to ../gcn_gate_classification/'s,
# repeated here so this file runs standalone (this repo's convention).
# ---------------------------------------------------------------------------
def matmul(A, B):
    rows, inner, cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def elementwise(A, B, op):
    return [[op(A[i][j], B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]


def hstack(A, B):
    return [A[i] + B[i] for i in range(len(A))]


def relu(A):
    return [[max(0.0, x) for x in row] for row in A]


def relu_grad_mask(Z):
    return [[1.0 if x > 0 else 0.0 for x in row] for row in Z]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Toy netlist builder — same shape as ../graph_representation_of_netlists/
# but parameterized so this file can build BOTH the 2-bit training netlist
# and an unseen 3-bit test netlist from one function, ripple-carry-style.
# ---------------------------------------------------------------------------
GATE_KINDS = ["PI", "AND2", "OR2", "XOR2", "DFF"]


def build_ripple_carry_adder(num_bits: int) -> dict:
    """Build an `num_bits`-wide ripple-carry adder netlist: for each bit i,
    xor_i/and_i(a/b)/sum_i/carry_i, chained through an external carry-in
    `cin` at bit 0 and carry(i-1) for every bit after — exactly the 2-bit
    pattern from ../graph_representation_of_netlists/ generalized to any
    width (that file's bit 0 is this function's `num_bits=2` case verbatim,
    node for node). Returns {"fanin": {...}, "kind": {...}}.
    """
    fanin: dict[str, list[str]] = {"cin": []}
    kind: dict[str, str] = {"cin": "PI"}
    for i in range(num_bits):
        for sig in (f"a{i}", f"b{i}"):
            fanin[sig] = []
            kind[sig] = "PI"
        cin = "cin" if i == 0 else f"carry{i - 1}"

        fanin[f"xor{i}"] = [f"a{i}", f"b{i}"]
        kind[f"xor{i}"] = "XOR2"
        fanin[f"and{i}a"] = [f"a{i}", f"b{i}"]
        kind[f"and{i}a"] = "AND2"
        fanin[f"sum{i}"] = [f"xor{i}", cin]
        kind[f"sum{i}"] = "XOR2"
        fanin[f"and{i}b"] = [f"xor{i}", cin]
        kind[f"and{i}b"] = "AND2"
        fanin[f"carry{i}"] = [f"and{i}a", f"and{i}b"]
        kind[f"carry{i}"] = "OR2"

        fanin[f"dff_sum{i}"] = [f"sum{i}"]
        kind[f"dff_sum{i}"] = "DFF"

    fanin[f"dff_cout{num_bits - 1}"] = [f"carry{num_bits - 1}"]
    kind[f"dff_cout{num_bits - 1}"] = "DFF"
    return {"fanin": fanin, "kind": kind}


def undirected_neighbors(netlist: dict) -> dict[str, list[str]]:
    """Every node's undirected neighbor list (both fan-in and fan-out),
    SORTED so `sample_neighbors` below is deterministic regardless of dict
    iteration order — important since we need reproducible runs to compare
    the training graph and the unseen test graph fairly.
    """
    neighbors: dict[str, set[str]] = {name: set() for name in netlist["fanin"]}
    for dst, srcs in netlist["fanin"].items():
        for src in srcs:
            neighbors[dst].add(src)
            neighbors[src].add(dst)
    return {name: sorted(ns) for name, ns in neighbors.items()}


def sample_neighbors(name: str, neighbors: dict[str, list[str]], k: int, seed: int) -> list[str]:
    """CONCEPT: the "Sample" in GraphSAGE. A node with more neighbors than k
    gets a random subset (a fixed-size sample keeps every forward pass the
    same cost regardless of how skewed the graph's fanout is — critical for
    GraphSAGE's original motivation, training on graphs too large to use
    every neighbor of a hub node). A node with k or fewer just uses all of
    them. `seed` is derived per-node so sampling is reproducible across
    repeated forward passes on the SAME graph, without needing to cache it.
    """
    candidates = neighbors[name]
    if len(candidates) <= k:
        return candidates
    return random.Random(seed ^ hash(name)).sample(candidates, k)


def build_feature_matrix(netlist: dict, node_ids: list[str]) -> list[list[float]]:
    fanout_count = {name: 0 for name in netlist["fanin"]}
    for srcs in netlist["fanin"].values():
        for src in srcs:
            fanout_count[src] += 1
    features = []
    for name in node_ids:
        row = [1.0 if netlist["kind"][name] == k else 0.0 for k in GATE_KINDS]
        row.append(len(netlist["fanin"][name]) / 2.0)
        row.append(min(fanout_count[name], 3) / 3.0)  # cap+normalize: the unseen graph has a fanout-3 node
        features.append(row)
    return features


def build_sample_matrix(node_ids: list[str], neighbors: dict[str, list[str]], k: int, seed: int) -> list[list[float]]:
    """S[v][u] = 1/|sampled(v)| if u is in v's sample, else 0 — a ROW-
    STOCHASTIC matrix (each row sums to 1, or to 0 for a node with no
    neighbors) so `S @ H` computes exactly `mean(h_u for u in sample(v))`
    per node in one matrix multiply, matching the aggregation step in the
    module docstring's formula.
    """
    index_of = {name: i for i, name in enumerate(node_ids)}
    n = len(node_ids)
    S = [[0.0] * n for _ in range(n)]
    for name in node_ids:
        sampled = sample_neighbors(name, neighbors, k, seed)
        if not sampled:
            continue
        weight = 1.0 / len(sampled)
        for nb in sampled:
            S[index_of[name]][index_of[nb]] = weight
    return S


# ---------------------------------------------------------------------------
# The 2-layer GraphSAGE encoder + a linear classifier head on top, trained
# together end to end. Structurally almost identical to ../gcn_gate_
# classification/'s GCN class — the ONLY difference is what gets multiplied
# by W: GCN multiplies the (self-looped, normalized) aggregation directly;
# SAGE concatenates the node's own features onto the aggregation first,
# then multiplies. That one change is what makes W's shape independent of
# any particular graph's size, which is the whole inductive story.
# ---------------------------------------------------------------------------
class GraphSAGE:
    def __init__(self, in_dim: int, hidden_dim: int, seed: int = 0):
        rng = random.Random(seed)
        s1 = math.sqrt(2.0 / (2 * in_dim))
        s2 = math.sqrt(2.0 / (2 * hidden_dim))
        sc = math.sqrt(2.0 / hidden_dim)
        self.W1 = [[rng.gauss(0, s1) for _ in range(hidden_dim)] for _ in range(2 * in_dim)]
        self.W2 = [[rng.gauss(0, s2) for _ in range(hidden_dim)] for _ in range(2 * hidden_dim)]
        self.Wc = [[rng.gauss(0, sc)] for _ in range(hidden_dim)]  # classifier head: embedding -> logit

    def forward(self, S: list[list[float]], X: list[list[float]]):
        agg0 = matmul(S, X)                    # layer 1 SAMPLE+AGGREGATE
        xcat0 = hstack(X, agg0)                 # concat(self, neighbor mean)
        z1 = matmul(xcat0, self.W1)
        h1 = relu(z1)                            # node embedding after 1 hop

        agg1 = matmul(S, h1)                    # layer 2 SAMPLE+AGGREGATE (now 2 hops from X)
        xcat1 = hstack(h1, agg1)
        z2 = matmul(xcat1, self.W2)
        h2 = relu(z2)                            # final node embedding

        zc = matmul(h2, self.Wc)                # classifier head
        p = [[sigmoid(zc[i][0])] for i in range(len(zc))]
        cache = {"X": X, "agg0": agg0, "xcat0": xcat0, "z1": z1, "h1": h1,
                 "agg1": agg1, "xcat1": xcat1, "z2": z2, "h2": h2}
        return p, h2, cache

    def backward(self, S, cache, p, y_mask, lr: float):
        n = len(p)
        labeled = [i for i in range(n) if y_mask[i] is not None]
        dzc = [[0.0] for _ in range(n)]
        for i in labeled:
            dzc[i][0] = (p[i][0] - y_mask[i]) / len(labeled)

        dWc = matmul(transpose(cache["h2"]), dzc)
        dh2 = matmul(dzc, transpose(self.Wc))
        dz2 = elementwise(dh2, relu_grad_mask(cache["z2"]), lambda a, b: a * b)
        dW2 = matmul(transpose(cache["xcat1"]), dz2)

        hidden = len(cache["h1"][0])
        dxcat1 = matmul(dz2, transpose(self.W2))
        dh1_self = [row[:hidden] for row in dxcat1]
        dagg1 = [row[hidden:] for row in dxcat1]
        dh1 = elementwise(dh1_self, matmul(transpose(S), dagg1), lambda a, b: a + b)  # self path + S^T-routed agg path
        dz1 = elementwise(dh1, relu_grad_mask(cache["z1"]), lambda a, b: a * b)
        dW1 = matmul(transpose(cache["xcat0"]), dz1)

        for W, dW in ((self.W1, dW1), (self.W2, dW2), (self.Wc, dWc)):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= lr * dW[i][j]


def bce_loss(p, y_mask) -> float:
    labeled = [i for i in range(len(p)) if y_mask[i] is not None]
    total = 0.0
    for i in labeled:
        pi = min(max(p[i][0], 1e-9), 1 - 1e-9)
        total += -(y_mask[i] * math.log(pi) + (1 - y_mask[i]) * math.log(1 - pi))
    return total / len(labeled)


def cosine_similarity(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u)) or 1e-9
    norm_v = math.sqrt(sum(a * a for a in v)) or 1e-9
    return dot / (norm_u * norm_v)


def main() -> None:
    K, SAMPLE_SEED = 2, 7

    # --- Train on the 2-bit adder ------------------------------------------------
    train_netlist = build_ripple_carry_adder(2)
    train_ids = list(train_netlist["fanin"].keys())
    train_index = {name: i for i, name in enumerate(train_ids)}
    train_neighbors = undirected_neighbors(train_netlist)
    S_train = build_sample_matrix(train_ids, train_neighbors, K, SAMPLE_SEED)
    X_train = build_feature_matrix(train_netlist, train_ids)

    # Same ground truth as ../gcn_gate_classification/: the carry chain.
    critical_train = {"xor0", "and0b", "carry0", "and1b", "carry1", "dff_cout1"}
    labeled_gates = ["xor0", "and0a", "sum0", "carry0", "dff_sum0", "xor1", "and1a", "and0b", "dff_sum1"]
    y_mask = [None] * len(train_ids)
    for gate in labeled_gates:
        y_mask[train_index[gate]] = 1.0 if gate in critical_train else 0.0

    sage = GraphSAGE(in_dim=len(X_train[0]), hidden_dim=6, seed=3)
    print(f"Training GraphSAGE on the 2-bit adder ({len(train_ids)} nodes, sample size k={K})...")
    for epoch in range(1, 401):
        p, h2, cache = sage.forward(S_train, X_train)
        sage.backward(S_train, cache, p, y_mask, lr=0.4)
        if epoch % 100 == 0:
            print(f"  epoch {epoch:>4}  loss = {bce_loss(p, y_mask):.4f}")

    # --- Zero-shot evaluation on an UNSEEN 3-bit adder ---------------------------
    # No retraining below this line — `sage.W1`, `sage.W2`, `sage.Wc` are frozen
    # exactly as trained above. Only the GRAPH changes.
    test_netlist = build_ripple_carry_adder(3)
    test_ids = list(test_netlist["fanin"].keys())
    test_index = {name: i for i, name in enumerate(test_ids)}
    test_neighbors = undirected_neighbors(test_netlist)
    S_test = build_sample_matrix(test_ids, test_neighbors, K, SAMPLE_SEED)
    X_test = build_feature_matrix(test_netlist, test_ids)

    print(f"\nUnseen 3-bit adder: {len(test_ids)} nodes (never seen during training).")
    print(f"carry1's undirected neighborhood in this graph has "
          f"{len(test_neighbors['carry1'])} nodes (and1a, and1b, sum2, and2b) — bigger than any "
          f"neighborhood in the 2-node-wide training graph, so k={K} sampling genuinely subsamples "
          "it here, unlike in training where every node had <= 2 neighbors.")

    p_test, h2_test, _ = sage.forward(S_test, X_test)
    critical_test = {"xor0", "and0b", "carry0", "and1b", "carry1", "and2b", "carry2", "dff_cout2"}

    print("\nZero-shot predictions on the unseen 3-bit adder (same trained weights, no fine-tuning):")
    correct = 0
    evaluated = [n for n in test_ids if test_netlist["kind"][n] != "PI"]
    for name in evaluated:
        i = test_index[name]
        pred = 1 if p_test[i][0] >= 0.5 else 0
        true = 1 if name in critical_test else 0
        correct += int(pred == true)
        flag, correct_flag = ("critical" if pred else "not critical"), ("correct" if pred == true else "WRONG")
        print(f"  {name:>11}: P(critical)={p_test[i][0]:.3f} -> {flag:<13} "
              f"(true={'critical' if true else 'not critical'}, {correct_flag})")
    print(f"\nZero-shot accuracy on the unseen 3-bit adder: {correct}/{len(evaluated)}")

    # A concrete embedding-space check: carry0 (trained on) and carry2 (never
    # seen) both sit on their respective circuit's carry chain and should end
    # up nearer each other in embedding space than carry0 is to an unrelated,
    # non-critical gate from the SAME unseen graph.
    carry0_emb = h2[train_index["carry0"]]
    carry2_emb = h2_test[test_index["carry2"]]
    and2a_emb = h2_test[test_index["and2a"]]  # non-critical gate in the unseen graph
    print(f"\ncosine_similarity(carry0 [trained], carry2 [unseen, critical])   = "
          f"{cosine_similarity(carry0_emb, carry2_emb):.3f}")
    print(f"cosine_similarity(carry0 [trained], and2a [unseen, NOT critical]) = "
          f"{cosine_similarity(carry0_emb, and2a_emb):.3f}")
    print(
        "\nA GCN (../gcn_gate_classification/) has no way to even ATTEMPT this: its A_hat is sized and\n"
        "shaped for exactly the 18-node training graph, so there's no forward pass to run on a 26-node\n"
        "graph at all without recomputing A_hat and retraining from scratch. GraphSAGE's W1/W2/Wc only\n"
        "ever operate on a fixed-size (self, neighbor-mean) vector, so the same weights are a function\n"
        "any node in any graph can be plugged into."
    )


if __name__ == "__main__":
    main()
