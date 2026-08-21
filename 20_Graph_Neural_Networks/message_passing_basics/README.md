# message_passing_basics

The core GNN mechanic — gather a message from each neighbor, then update — illustrated by propagating a toggle-rate-like activity value between neighboring gates over multiple hops.

## message_passing_basics.py

Spreads a toggle-rate value from the toy adder's primary inputs through the netlist, round by round, then keeps running rounds well past the point where it's useful to show real over-smoothing.

### Concepts covered

- **`message_passing_round`** — one gather-then-update round over DIRECTED fan-in only: a node's new value is the mean of its fan-in neighbors' current values. This is message passing with no learned weights at all — everything after this file (`../gcn_gate_classification/`, `../graphsage_inductive_netlist_embedding/`, `../gat_attention_critical_path/`) is a variation on this same gather/update shape, made learnable.
- **Receptive field grows with depth** — `cin`'s low toggle rate doesn't reach `carry0` until round 2 (2 hops away) and doesn't reach `dff_cout` until round 5. A K-layer GNN can only "see" K hops of graph structure, no matter how much training data it gets — this is the concrete demonstration of why real GNN stacks stay shallow (2-4 layers).
- **`message_passing_round_undirected`** — the aggregation shape GCN layers actually use (undirected, self-looped neighborhoods, see `../gcn_gate_classification/`'s `build_normalized_adjacency`), as opposed to the directed fan-in-only version above.
- **Real over-smoothing** — the directed version reaches a stable fixed point and stops changing (harmless: a DAG has no cycles to keep re-mixing values). The undirected version genuinely converges every node toward the SAME value after enough rounds — a random walk toward the graph's single stationary distribution — which is the actual failure mode real GCN-style architectures guard against by staying shallow.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/message_passing_basics/message_passing_basics.py
```

```
round           cin       and0b      carry0       and1b      carry1    dff_cout
-------------------------------------------------------------------------------
0             0.100       0.000       0.000       0.000       0.000       0.000
1             0.100       0.050       0.000       0.000       0.000       0.000
2             0.100       0.500       0.475       0.450       0.450       0.000
...
```

### Configuration

- `NETLIST_FANIN` — the same toy 2-bit adder as `../graph_representation_of_netlists/`, condensed to just fanin lists
- `main()`'s `initial` dict — each primary input's starting toggle rate; try making `cin` toggle as often as the data inputs and watch the carry chain's distinctiveness disappear even faster

### See also

- `../graph_representation_of_netlists/README.md` — the graph this file propagates values over
- `../gcn_gate_classification/README.md` — the same undirected aggregation shape, made learnable with a weight matrix
