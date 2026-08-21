# gnn_applications_eda

A survey of real EDA/RTL use cases for GNNs — congestion/routability prediction, timing prediction, power estimation, and bug/anomaly detection — each illustrated with a minimal, runnable synthetic example.

## gnn_applications_eda.py

Four short, independent sections, each reusing the same toy 2-bit adder netlist and a variation on the message-passing mechanic from `../message_passing_basics/`, rather than re-deriving a full trained model (that machinery already lives in the other templates in this directory).

### Concepts covered

- **`congestion_prediction`** — a synthetic placement (each gate assigned an (x, y) grid cell) plus edge-counting per cell approximates what a routability-prediction GNN estimates from a placed netlist graph before running a full, slow router. The hottest cell lands exactly where `carry0` and its neighbors sit — the same carry-chain hot spot every other template in this directory flags structurally.
- **`timing_prediction`** — message passing with `max` as the aggregation function instead of `mean`/`sum`, propagating per-gate delays forward through the netlist to compute each node's arrival time — the same core operation real Static Timing Analysis performs, and what a timing-prediction GNN learns to approximate directly from graph structure.
- **`power_estimation`** — reuses `../message_passing_basics/`'s toggle-rate propagation directly, then turns activity into a power estimate via `activity x capacitance_proxy` (fanout count standing in for real wire/load capacitance).
- **`anomaly_detection`** — injects one deliberately wrong gate (a 3-input AND spliced into a technology library where every other gate is 2-input) and computes a RAW per-node z-score alongside an AGGREGATED (mean-pooled with undirected neighbors) z-score, showing that a graph-aware check surfaces where a bug's effect concentrates structurally — not always the buggy node itself.
- **Same mechanic, different target** — all four sections aggregate structural or physical neighborhood information and turn the aggregate into a per-node prediction; what changes is only WHAT gets aggregated and WHAT it predicts.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 20_Graph_Neural_Networks/gnn_applications_eda/gnn_applications_eda.py
```

No `ANTHROPIC_API_KEY` needed.

```
======================================
1. Congestion / routability prediction
======================================
Estimated wire count per grid cell (higher = more congested):
  (0,0)= 4  (0,1)= 2  (0,2)= 4
  (1,0)=14  (1,1)=12  (1,2)=11
  (2,0)= 1  (2,1)=12  (2,2)= 9

Hottest cell: (1, 0) with 14 wires ...
```

### Configuration

- `congestion_prediction`'s `placement` dict — the synthetic (row, col) grid position for each gate
- `timing_prediction`'s `delay_of_kind` — illustrative per-gate-kind delays, not a real calibrated cell library
- `power_estimation`'s `capacitance_proxy` — fanout count as a stand-in for real wire/load capacitance
- `anomaly_detection`'s injected bug (`fanin["and0a"] = [...]`) and the `1.5` z-score flagging threshold

### See also

- `../message_passing_basics/README.md` — the aggregation mechanic every section here is a variation of
- `../gcn_gate_classification/README.md` — a fully trained version of the same "structure predicts a per-node property" idea this file only sketches
