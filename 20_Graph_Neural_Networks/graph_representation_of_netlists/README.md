# graph_representation_of_netlists

How an RTL netlist (gates, flip-flops, wires) becomes a graph — nodes = gates/ports, edges = wires/connections — before any GNN can run over it.

## graph_representation_of_netlists.py

Builds a graph out of a toy 2-bit ripple-carry adder netlist (18 nodes: 5 primary inputs, 13 gates/flip-flops), prints a full summary, and saves it to `netlist_graph.json`. This adder is the running example for every template in this topic directory.

### Concepts covered

- **Nodes and directed edges instead of Verilog** — a gate becomes a node; a wire becomes a directed edge from the driving gate's output to the driven gate's input. Contrast with `../../09_RAG_and_Knowledge/graph_rag/graph_construction.py`, which builds a graph of a SOFTWARE codebase (modules/classes/functions) using the same nodes-dict + edges-list shape, but over a completely different domain.
- **`_one_hot`** — turning a gate's categorical `kind` (PI/AND2/OR2/XOR2/DFF) into a numeric vector, the same trick embeddings use to turn tokens into vectors.
- **`build_feature_matrix`** — the numeric `X` matrix a GNN layer actually multiplies against: one-hot gate kind plus fanin_count/fanout_count, cheap structural features that need no simulation to compute.
- **Edge direction matters** — a wire has a driver and a receiver; edges always point driver -> receiver. This directionality is what makes message passing (`../message_passing_basics/`) propagate the way real signals actually flow.
- **`carry0`** — the one signal that crosses from bit 0's logic into bit 1's, giving it the highest fanout among combinational gates and putting it on the circuit's longest path. Every other template in this directory uses `carry0`'s special role as the running example.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/graph_representation_of_netlists/graph_representation_of_netlists.py
```

No `ANTHROPIC_API_KEY` needed — this whole topic directory is pure graph/numeric code, no API calls.

```
18 nodes: 5 PI, 4 XOR2, 4 AND2, 2 OR2, 3 DFF
23 directed edges (wires)

Wires (driver --> receiver):
          a0 --> xor0
          ...
```

### Configuration

- `NETLIST` — the toy netlist, described as fanin lists per gate (the way a real netlist parser would see a `.v` file's structural instances)
- `GATE_KINDS` — the fixed vocabulary one-hot encoded into node features
- `GRAPH_FILE` — where the graph is persisted (`netlist_graph.json`, gitignored — a runtime artifact, not source)

### See also

- `../message_passing_basics/README.md` — the first thing done WITH this graph representation
- `../../09_RAG_and_Knowledge/graph_rag/README.md` — the same nodes/edges idea applied to a software codebase instead of hardware
