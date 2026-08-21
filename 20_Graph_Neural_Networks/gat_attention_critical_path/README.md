# gat_attention_critical_path

Using a Graph Attention Network (GAT) to learn per-neighbor attention weights, showing which fan-in signal actually matters for a gate's role on the timing-critical path.

## gat_attention_critical_path.py

Trains a 2-layer GAT (with manually-derived backprop through the attention softmax) on the same critical-path classification task as `../gcn_gate_classification/`, then inspects `carry1`'s learned attention over its two fan-in gates.

### Concepts covered

- **Learned, per-neighbor weights instead of a fixed average** — `e_vu = LeakyReLU(a . [W h_v || W h_u])`, softmax-normalized into `alpha_vu`, then `h_v' = ReLU(sum_u alpha_vu * W h_u)`. Unlike GCN's fixed degree-normalized average (`../gcn_gate_classification/`) or GraphSAGE's random sample (`../graphsage_inductive_netlist_embedding/`), a GAT layer decides how much each neighbor matters, per node, per forward pass.
- **Attention over directed fan-in, not the undirected neighborhood** — `NEIGHBOR_SETS` is each node's fan-in plus itself (self-loop), the natural framing for "which of my INPUTS matters," as opposed to the undirected shape the other trainable templates use.
- **The headline result: `and1a` vs. `and1b`** — two structurally near-identical AND2 gates; the only thing that tells them apart is that `and1b`'s own fan-in includes `carry0`, the carry-chain node. `alpha(carry1 <- and1b)` comes out clearly higher than `alpha(carry1 <- and1a)` after training — the same judgment a timing engineer would make by hand, discovered purely from labels.
- **Why LAYER 2's attention, not layer 1's** — `and1a`/`and1b` have identical RAW features, so layer 1's attention over raw `X` can't distinguish them (and comes out ~equal in the code's own printed run). It's only after layer 1 that `and1b`'s embedding has absorbed `carry0`'s signal through its own fan-in aggregation — layer 2's attention, operating on those embeddings, is where the distinguishing signal is actually visible.
- **`GATLayer.forward` / `.backward`** — per-node loops (not whole-graph matrix multiplies, since neighbor counts and identities differ per node) implementing the softmax-attention forward pass and its manually-derived analytic gradient, verified against a finite-difference check during development.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/gat_attention_critical_path/gat_attention_critical_path.py
```

No `ANTHROPIC_API_KEY` needed.

```
Training 2-layer GAT on 7 labeled gates...
  epoch  100  loss = 0.3298
  ...
Test accuracy: 4/6

carry1's fan-in attention weights, LAYER 2 (attention over h1 embeddings, ...):
  alpha(carry1 <-  carry1) = 0.766  <- self
  alpha(carry1 <-   and1a) = 0.000  <- feeds from a1/b1 only
  alpha(carry1 <-   and1b) = 0.234  <- feeds from carry0 (critical)
```

### Configuration

- `GATLayer(in_dim=..., hidden_dim=6, rng=...)` — attention head width; this file uses a single attention head, real GAT implementations typically use several in parallel and concatenate their outputs
- The training loop's `lr=0.3` / `401` epochs, `random.Random(11)` seed

### See also

- `../gcn_gate_classification/README.md` — the same critical-path task with fixed, degree-normalized aggregation instead of learned attention
- `../message_passing_basics/README.md` — the un-weighted gather/update mechanic every aggregation scheme in this directory (including attention) builds on
