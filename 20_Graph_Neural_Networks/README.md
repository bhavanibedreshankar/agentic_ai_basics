# Graph_Neural_Networks

Graph Neural Networks (GNNs), taught through RTL/hardware design: an RTL netlist is already a graph — gates and flip-flops are nodes, wires are edges — so every mechanic here (message passing, GCN, GraphSAGE, GAT) is demonstrated directly on a toy 2-bit ripple-carry adder netlist instead of a generic social-network or citation-graph example. Six templates, each in its own directory, going from the smallest mechanic (representing a netlist as a graph) up through three learnable GNN architectures to a survey of real EDA use cases.

Unlike the rest of this repo, none of these templates call the Claude API — they're pure, dependency-free Python (plain lists/loops, no numpy or PyTorch), so nothing here needs `ANTHROPIC_API_KEY`. Every trainable template (`gcn_gate_classification`, `graphsage_inductive_netlist_embedding`, `gat_attention_critical_path`) implements its own forward pass and analytic backpropagation by hand, verified with a finite-difference gradient check during development.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`graph_representation_of_netlists/`](graph_representation_of_netlists/README.md) | How a toy 2-bit ripple-carry adder netlist becomes a graph (nodes = gates/ports, edges = wires) — the representation every other template in this directory builds on |
| 2 | [`message_passing_basics/`](message_passing_basics/README.md) | The core gather-then-update mechanic, with no learned weights at all: propagating a toggle-rate value hop by hop, and a genuine demonstration of over-smoothing |
| 3 | [`gcn_gate_classification/`](gcn_gate_classification/README.md) | A Graph Convolutional Network: message passing plus a learned weight matrix, classifying which gates sit on the timing-critical path — transductive, tied to one fixed graph |
| 4 | [`graphsage_inductive_netlist_embedding/`](graphsage_inductive_netlist_embedding/README.md) | GraphSAGE: sampled local neighbor aggregation instead of a global matrix, so the trained weights transfer, zero-shot, to a completely unseen netlist |
| 5 | [`gat_attention_critical_path/`](gat_attention_critical_path/README.md) | A Graph Attention Network: learned, per-neighbor attention weights instead of a fixed average, discovering which fan-in signal actually matters for timing |
| 6 | [`gnn_applications_eda/`](gnn_applications_eda/README.md) | A survey of four real EDA use cases (congestion prediction, timing prediction, power estimation, bug/anomaly detection), each a small runnable variation on the message-passing mechanic |

## Setup

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
```

No `ANTHROPIC_API_KEY` needed anywhere in this topic directory.

Run any template from the repo root, e.g.:

```bash
python3 20_Graph_Neural_Networks/graph_representation_of_netlists/graph_representation_of_netlists.py
```

## The running example

Every template in this directory uses the same toy circuit: a 2-bit ripple-carry adder computing `SUM = A + B + CIN` (5 primary inputs, 13 gates/flip-flops, 18 nodes total). `carry0` — the one signal that crosses from bit 0's logic into bit 1's — sits on the circuit's longest combinational path and has the highest fanout among combinational gates, making it the recurring example of "what a GNN can learn to notice" across message passing, GCN classification, GraphSAGE embeddings, GAT attention, and EDA congestion/anomaly detection alike. `graphsage_inductive_netlist_embedding/` additionally builds a 3-bit version of the same adder to demonstrate generalization to an unseen graph.

## How these relate to each other

| Directory | Learned weights? | Aggregation | Works on an unseen graph? |
|---|---|---|---|
| `message_passing_basics/` | No | Fixed mean (directed or undirected) | N/A — no training |
| `gcn_gate_classification/` | Yes | Fixed, degree-normalized (whole graph) | No — tied to one `A_hat` matrix |
| `graphsage_inductive_netlist_embedding/` | Yes | Learned function over a random neighbor sample | Yes — proven directly on an unseen 3-bit adder |
| `gat_attention_critical_path/` | Yes | Learned, per-neighbor attention weights | In principle (local function, like SAGE) — not demonstrated here |
| `gnn_applications_eda/` | No (illustrative only) | Varies per section | N/A — survey, not a trained model |

## Advantages and disadvantages (recap)

GNNs handle graph-structured data (like a netlist) that CNNs/RNNs can't represent naturally, are permutation-invariant, and share parameters across every node regardless of graph size. The costs, made concrete in this directory: over-smoothing with too many layers (`message_passing_basics/`), a shallow effective receptive field (2-4 hops is typical, same file), and a transductive architecture's inability to generalize to a new graph at all (`gcn_gate_classification/`, contrasted directly with `graphsage_inductive_netlist_embedding/`'s inductive alternative).
