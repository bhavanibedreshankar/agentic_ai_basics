# gcn_gate_classification

Using a Graph Convolutional Network (GCN) to classify gates in a netlist — predicting whether a gate sits on the timing-critical path — from structural features plus 2 hops of graph structure.

## gcn_gate_classification.py

Trains a 2-layer GCN, from scratch with manually-derived backpropagation, on 7 of the toy adder's 13 gates, then evaluates it on the 6 held-out gates it never saw a label for.

### Concepts covered

- **A GCN layer = message passing (`../message_passing_basics/`) + a learned weight matrix + ReLU** — `H' = ReLU(A_hat @ H @ W)`, where `A_hat` is the symmetrically-normalized, self-looped adjacency matrix built by `build_normalized_adjacency` (Kipf & Welling's degree normalization, so high-fanout nodes like `carry0` don't dominate every neighbor's update just by touching more edges).
- **Semi-supervised, TRANSDUCTIVE learning** — only `LABELED_GATES` contribute to the training loss; `TEST_GATES` are predicted using nothing but the learned weights plus 2-hop structure. Message passing still runs over the WHOLE graph (including unlabeled and PI nodes) — this is the classic Kipf & Welling GCN setup, and the opposite of `../graphsage_inductive_netlist_embedding/`, which never needs a fixed, whole-graph matrix at all.
- **`GCN.forward` / `GCN.backward`** — a from-scratch 2-layer forward pass and manually-derived analytic backprop (matrix calculus, not autograd), verified against a finite-difference gradient check during development (see the module docstring's note).
- **Depth as a feature** — `_logic_depth` adds each gate's longest path from a primary input as a third structural feature; depth alone doesn't fully separate critical from non-critical gates (two gates at the same depth can have different labels), so the task genuinely needs graph structure, not just a per-node lookup.
- **and1a vs and1b** — two structurally near-identical AND2 gates with the same fanin/fanout counts; the model gets one right and the other wrong in the default run, an honest demonstration that a tiny 18-node graph with 7 labels is a small, noisy training signal, same as a real semi-supervised GCN with limited labeled data.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/gcn_gate_classification/gcn_gate_classification.py
```

No `ANTHROPIC_API_KEY` needed.

```
Training on 7 labeled gates: ['xor0', 'carry0', 'dff_cout', 'and0a', 'sum0', 'xor1', 'dff_sum1']
Held out (never shown a label) 6 test gates: ['and0b', 'dff_sum0', 'and1a', 'sum1', 'and1b', 'carry1']

epoch    1  loss = 0.7267
...
epoch  400  loss = 0.4506

Final predictions on held-out TEST gates (never used in the loss):
       and0b: P(critical)=0.515 -> critical      (true=critical, correct)
       ...
Test accuracy: 5/6
```

### Configuration

- `LABELED_GATES` / `TEST_GATES` — the semi-supervised split; chosen to keep both the critical/non-critical class balance and each test gate's graph distance to the nearest label roughly even
- `GCN(in_dim=..., hidden_dim=6, seed=38)` and the training loop's `lr=0.3` / `401` epochs — tuned by a small grid search across seeds/learning rates (see the module for the exact numbers); this is a genuinely small, noisy setup, so results are seed-sensitive

### See also

- `../message_passing_basics/README.md` — the ungated aggregation mechanic this file adds a learned weight matrix on top of
- `../graphsage_inductive_netlist_embedding/README.md` — the inductive alternative: a GCN's `A_hat` can't be reused on a different graph at all, GraphSAGE's local aggregation function can
