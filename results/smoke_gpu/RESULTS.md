# SMOKE CHECK ONLY - NOT THESIS RESULTS

Study: main; completed cells: 8/8.

Values are mean +/- sample SD of seed means. JSON/CSV include 95% t intervals.
Asset protection means the attacker did not reach its configured objective before the horizon.
Dwell counts decoy-engaged steps; depth counts distinct technique IDs engaged by decoys.

## Six model combinations

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 3.250 (SD unavailable) | 2.750 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| GRU + IPPO | 3.750 (SD unavailable) | 3.500 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| LSTM + IPPO | 3.625 (SD unavailable) | 3.125 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| Transformer + MAPPO | 2.625 (SD unavailable) | 2.375 (SD unavailable) | 25.000 (SD unavailable)% | 1 |
| GRU + MAPPO | 2.625 (SD unavailable) | 2.375 (SD unavailable) | 25.000 (SD unavailable)% | 1 |
| LSTM + MAPPO | 3.375 (SD unavailable) | 2.875 (SD unavailable) | 37.500 (SD unavailable)% | 1 |

## No-intent controls

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| NoHistory + IPPO | 3.375 (SD unavailable) | 2.875 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 2.500 (SD unavailable) | 2.250 (SD unavailable) | 25.000 (SD unavailable)% | 1 |

## Requested ablation

Validation-selected encoder: **Transformer**.
Selection uses mean validation dwell only; test outcomes are not used for selection.

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 3.250 (SD unavailable) | 2.750 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| NoHistory + IPPO | 3.375 (SD unavailable) | 2.875 (SD unavailable) | 37.500 (SD unavailable)% | 1 |
| Transformer + MAPPO | 2.625 (SD unavailable) | 2.375 (SD unavailable) | 25.000 (SD unavailable)% | 1 |

The first two rows isolate intent under IPPO; first versus third isolates critic information.
NoHistory + MAPPO above additionally isolates intent under MAPPO.
These selected-model ablations are descriptive/exploratory. Non-significance does not prove equivalence.

## Scenario S1

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| GRU + IPPO | 5.000 (SD unavailable) | 5.000 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| LSTM + IPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| Transformer + MAPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| GRU + MAPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| LSTM + MAPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| NoHistory + IPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 5.500 (SD unavailable) | 4.500 (SD unavailable) | 50.000 (SD unavailable)% | 1 |

## Scenario S2

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 6.000 (SD unavailable) | 4.000 (SD unavailable) | 100.000 (SD unavailable)% | 1 |
| GRU + IPPO | 8.000 (SD unavailable) | 6.000 (SD unavailable) | 100.000 (SD unavailable)% | 1 |
| LSTM + IPPO | 7.000 (SD unavailable) | 5.000 (SD unavailable) | 100.000 (SD unavailable)% | 1 |
| Transformer + MAPPO | 1.000 (SD unavailable) | 1.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| GRU + MAPPO | 1.000 (SD unavailable) | 1.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| LSTM + MAPPO | 7.000 (SD unavailable) | 5.000 (SD unavailable) | 100.000 (SD unavailable)% | 1 |
| NoHistory + IPPO | 7.000 (SD unavailable) | 5.000 (SD unavailable) | 100.000 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 1.000 (SD unavailable) | 1.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |

## Scenario S3

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| GRU + IPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| LSTM + IPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| Transformer + MAPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| GRU + MAPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| LSTM + MAPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| NoHistory + IPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |

## Scenario S6

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| GRU + IPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| LSTM + IPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| Transformer + MAPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| GRU + MAPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| LSTM + MAPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| NoHistory + IPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 0.000 (SD unavailable) | 0.000 (SD unavailable) | 0.000 (SD unavailable)% | 1 |

## Scenario S7

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 2.333 (SD unavailable) | 2.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| GRU + IPPO | 3.333 (SD unavailable) | 3.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| LSTM + IPPO | 3.000 (SD unavailable) | 3.000 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| Transformer + MAPPO | 2.333 (SD unavailable) | 2.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| GRU + MAPPO | 2.333 (SD unavailable) | 2.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| LSTM + MAPPO | 2.333 (SD unavailable) | 2.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| NoHistory + IPPO | 2.333 (SD unavailable) | 2.333 (SD unavailable) | 33.333 (SD unavailable)% | 1 |
| NoHistory + MAPPO | 2.000 (SD unavailable) | 2.000 (SD unavailable) | 33.333 (SD unavailable)% | 1 |

## Limits

- Fixed five-zone engagement simulator, with four parameter-sharing agents.
- Synthetic Markov episodes estimated from verified CAM-LDS playbooks, not raw-log replay or live attacks.
- NoHistory still observes local telemetry and recent locally observed technique; it removes the shared history embedding.
- Reward optimizes dwell only. Depth and asset protection are evaluated outcomes, not guaranteed improvements.
- Architecture choices differ from historical v7; old checkpoints and reported scores are not reused.
- A small smoke run verifies execution only. It does not measure learning quality or support thesis claims.
