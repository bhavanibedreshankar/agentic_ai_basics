# graphsage_inductive_netlist_embedding

Using GraphSAGE-style neighbor sampling and aggregation to generate netlist gate embeddings, then applying the exact same trained weights — zero retraining — to a completely unseen netlist.

## graphsage_inductive_netlist_embedding.py

Trains a 2-layer GraphSAGE encoder plus a linear classifier head on the toy 2-bit adder, then evaluates it, unchanged, on a freshly-built 3-bit adder it has never seen.

### Concepts covered

- **Local aggregation instead of a global matrix** — a GraphSAGE layer is `h_v' = ReLU(W . concat(h_v, mean(sample(neighbors(v), k))))`. Unlike `../gcn_gate_classification/`'s GCN, which bakes one fixed, whole-graph `A_hat` matrix into every forward pass, this function only ever looks at a node's own features and a sampled subset of its immediate neighbors — nothing about `W`'s shape depends on graph size.
- **`build_ripple_carry_adder(num_bits)`** — generalizes the toy 2-bit adder from `../graph_representation_of_netlists/` to any width, so the exact same construction logic builds both the training graph (2 bits) and a structurally-different, never-seen test graph (3 bits, 26 nodes vs. 18, with one node — `carry1` — that has a bigger neighborhood than anything seen during training).
- **`sample_neighbors`** — the "Sample" in GraphSAGE: a node with more neighbors than `k` gets a random, fixed-size subset, keeping every forward pass the same cost regardless of how skewed a real netlist's fanout is.
- **`build_sample_matrix`** — turns that sampling rule into a row-stochastic matrix `S` so `S @ H` computes the mean-of-sampled-neighbors step in one matrix multiply, letting `GraphSAGE.forward`/`.backward` reuse the same matrix-calculus shape as `../gcn_gate_classification/`'s GCN.
- **Zero-shot transfer, proven two ways** — 18/19 classification accuracy on the unseen 3-bit adder's gates with NO fine-tuning, and a direct embedding-space check: `cosine_similarity(carry0 [trained], carry2 [unseen, critical])` comes out near 1.0, versus near 0 against an unrelated, non-critical gate from the same unseen graph.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/graphsage_inductive_netlist_embedding/graphsage_inductive_netlist_embedding.py
```

No `ANTHROPIC_API_KEY` needed.

```
Training GraphSAGE on the 2-bit adder (18 nodes, sample size k=2)...
  epoch  100  loss = 0.1542
  ...
Unseen 3-bit adder: 26 nodes (never seen during training).
...
Zero-shot accuracy on the unseen 3-bit adder: 18/19

cosine_similarity(carry0 [trained], carry2 [unseen, critical])   = 1.000
cosine_similarity(carry0 [trained], and2a [unseen, NOT critical]) = 0.094
```

### Configuration

- `K` (sample size) / `SAMPLE_SEED` — how many neighbors get sampled and the seed controlling which ones, when a node has more than `K`
- `GraphSAGE(in_dim=..., hidden_dim=6, seed=3)` and the training loop's `lr=0.4` / `401` epochs

### See also

- `../gcn_gate_classification/README.md` — the transductive counterpart: same critical-path task, but the `A_hat` matrix can't transfer to a different graph at all
- `../graph_representation_of_netlists/README.md` — the same 2-bit adder construction this file's `build_ripple_carry_adder(2)` reproduces exactly
