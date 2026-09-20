# Planned results (full experiment not run)

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection |
|---|---:|---:|---:|
| Transformer + IPPO | Pending | Pending | Pending |
| GRU + IPPO | Pending | Pending | Pending |
| LSTM + IPPO | Pending | Pending | Pending |
| Transformer + MAPPO | Pending | Pending | Pending |
| GRU + MAPPO | Pending | Pending | Pending |
| LSTM + MAPPO | Pending | Pending | Pending |

Best MAPPO encoder is selected on validation dwell after all main cells finish.

| Ablation | Dwell time (steps) | Interaction depth (techniques) | Asset protection |
|---|---:|---:|---:|
| Selected encoder + IPPO | Pending | Pending | Pending |
| NoHistory + IPPO | Pending | Pending | Pending |
| Selected encoder + MAPPO | Pending | Pending | Pending |
| NoHistory + MAPPO (additional control) | Pending | Pending | Pending |

The same three metrics are reported per scenario and in the optional LOSO study.
Actual reports include mean, sample SD, number of seeds, and JSON/CSV confidence
intervals. No historical result is inserted into these tables.
