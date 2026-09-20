# Thesis version 2310 results

Study: main; completed cells: 80/80.

Values are mean +/- sample SD of seed means. JSON/CSV include 95% t intervals.
Asset protection means the attacker did not reach its configured objective before the horizon.
Dwell counts decoy-engaged steps; depth counts distinct technique IDs engaged by decoys.

## Six model combinations

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 8.080 +/- 0.403 | 5.556 +/- 0.186 | 46.160 +/- 2.781% | 10 |
| GRU + IPPO | 7.998 +/- 0.441 | 5.505 +/- 0.230 | 48.300 +/- 4.356% | 10 |
| LSTM + IPPO | 8.186 +/- 0.488 | 5.577 +/- 0.280 | 46.580 +/- 4.006% | 10 |
| Transformer + MAPPO | 8.612 +/- 0.369 | 5.832 +/- 0.204 | 49.700 +/- 2.649% | 10 |
| GRU + MAPPO | 8.445 +/- 0.579 | 5.746 +/- 0.354 | 47.760 +/- 3.621% | 10 |
| LSTM + MAPPO | 8.457 +/- 0.324 | 5.752 +/- 0.217 | 48.580 +/- 4.242% | 10 |

## No-intent controls

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| NoHistory + IPPO | 6.262 +/- 0.351 | 4.549 +/- 0.248 | 36.080 +/- 3.213% | 10 |
| NoHistory + MAPPO | 6.568 +/- 0.397 | 4.699 +/- 0.257 | 35.540 +/- 3.586% | 10 |

## Requested ablation

Validation-selected encoder: **Transformer**.
Selection uses mean validation dwell only; test outcomes are not used for selection.

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 8.080 +/- 0.403 | 5.556 +/- 0.186 | 46.160 +/- 2.781% | 10 |
| NoHistory + IPPO | 6.262 +/- 0.351 | 4.549 +/- 0.248 | 36.080 +/- 3.213% | 10 |
| Transformer + MAPPO | 8.612 +/- 0.369 | 5.832 +/- 0.204 | 49.700 +/- 2.649% | 10 |

The first two rows isolate intent under IPPO; first versus third isolates critic information.
NoHistory + MAPPO above additionally isolates intent under MAPPO.
These selected-model ablations are descriptive/exploratory. Non-significance does not prove equivalence.

## Scenario S1

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 15.557 +/- 1.054 | 9.775 +/- 0.461 | 51.455 +/- 6.548% | 10 |
| GRU + IPPO | 15.571 +/- 1.133 | 9.738 +/- 0.591 | 52.125 +/- 5.467% | 10 |
| LSTM + IPPO | 15.960 +/- 0.910 | 9.929 +/- 0.554 | 52.918 +/- 5.112% | 10 |
| Transformer + MAPPO | 16.201 +/- 0.968 | 9.899 +/- 0.631 | 53.403 +/- 6.594% | 10 |
| GRU + MAPPO | 16.508 +/- 0.954 | 10.160 +/- 0.498 | 54.314 +/- 6.063% | 10 |
| LSTM + MAPPO | 16.207 +/- 1.316 | 10.057 +/- 0.586 | 53.040 +/- 6.397% | 10 |
| NoHistory + IPPO | 13.370 +/- 0.595 | 9.129 +/- 0.450 | 46.144 +/- 5.196% | 10 |
| NoHistory + MAPPO | 14.465 +/- 1.539 | 9.613 +/- 0.644 | 49.721 +/- 7.643% | 10 |

## Scenario S2

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 10.808 +/- 1.214 | 6.870 +/- 0.775 | 65.989 +/- 12.557% | 10 |
| GRU + IPPO | 10.918 +/- 1.096 | 6.940 +/- 0.591 | 70.093 +/- 7.628% | 10 |
| LSTM + IPPO | 11.011 +/- 0.823 | 7.003 +/- 0.562 | 66.935 +/- 8.003% | 10 |
| Transformer + MAPPO | 11.668 +/- 0.795 | 7.321 +/- 0.532 | 71.605 +/- 8.736% | 10 |
| GRU + MAPPO | 11.443 +/- 0.888 | 7.201 +/- 0.672 | 71.559 +/- 8.348% | 10 |
| LSTM + MAPPO | 11.452 +/- 0.838 | 7.144 +/- 0.619 | 71.265 +/- 9.689% | 10 |
| NoHistory + IPPO | 9.198 +/- 0.676 | 5.932 +/- 0.605 | 52.828 +/- 6.400% | 10 |
| NoHistory + MAPPO | 9.355 +/- 0.874 | 5.978 +/- 0.541 | 53.462 +/- 8.231% | 10 |

## Scenario S3

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 8.626 +/- 0.730 | 6.703 +/- 0.574 | 37.782 +/- 5.707% | 10 |
| GRU + IPPO | 8.241 +/- 1.476 | 6.393 +/- 1.020 | 35.381 +/- 10.948% | 10 |
| LSTM + IPPO | 8.309 +/- 0.962 | 6.568 +/- 0.720 | 35.192 +/- 9.965% | 10 |
| Transformer + MAPPO | 9.168 +/- 1.562 | 7.137 +/- 1.100 | 41.612 +/- 8.409% | 10 |
| GRU + MAPPO | 8.671 +/- 1.381 | 6.838 +/- 1.184 | 38.884 +/- 12.155% | 10 |
| LSTM + MAPPO | 8.860 +/- 0.919 | 6.900 +/- 0.743 | 41.592 +/- 8.336% | 10 |
| NoHistory + IPPO | 4.110 +/- 0.985 | 3.473 +/- 0.810 | 18.374 +/- 7.037% | 10 |
| NoHistory + MAPPO | 4.854 +/- 1.528 | 4.067 +/- 1.269 | 21.031 +/- 13.874% | 10 |

## Scenario S4

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 6.792 +/- 0.896 | 5.008 +/- 0.484 | 40.985 +/- 7.916% | 10 |
| GRU + IPPO | 6.869 +/- 0.717 | 5.047 +/- 0.508 | 43.208 +/- 6.843% | 10 |
| LSTM + IPPO | 6.720 +/- 0.472 | 4.966 +/- 0.348 | 41.037 +/- 5.787% | 10 |
| Transformer + MAPPO | 6.800 +/- 0.936 | 5.024 +/- 0.536 | 39.697 +/- 9.198% | 10 |
| GRU + MAPPO | 6.866 +/- 0.746 | 5.111 +/- 0.463 | 41.333 +/- 8.577% | 10 |
| LSTM + MAPPO | 6.980 +/- 0.615 | 5.178 +/- 0.485 | 41.971 +/- 7.136% | 10 |
| NoHistory + IPPO | 5.669 +/- 0.750 | 4.432 +/- 0.445 | 28.143 +/- 10.713% | 10 |
| NoHistory + MAPPO | 6.050 +/- 0.595 | 4.650 +/- 0.357 | 29.013 +/- 7.929% | 10 |

## Scenario S5

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 1.140 +/- 0.415 | 1.073 +/- 0.366 | 56.771 +/- 9.409% | 10 |
| GRU + IPPO | 1.439 +/- 0.540 | 1.338 +/- 0.463 | 66.428 +/- 16.032% | 10 |
| LSTM + IPPO | 1.112 +/- 0.478 | 1.051 +/- 0.421 | 54.842 +/- 14.815% | 10 |
| Transformer + MAPPO | 1.307 +/- 0.486 | 1.209 +/- 0.399 | 60.604 +/- 14.568% | 10 |
| GRU + MAPPO | 1.049 +/- 0.517 | 0.992 +/- 0.446 | 52.538 +/- 15.638% | 10 |
| LSTM + MAPPO | 1.047 +/- 0.396 | 0.998 +/- 0.337 | 53.015 +/- 13.237% | 10 |
| NoHistory + IPPO | 0.892 +/- 0.418 | 0.859 +/- 0.365 | 48.366 +/- 14.013% | 10 |
| NoHistory + MAPPO | 0.700 +/- 0.102 | 0.690 +/- 0.097 | 40.984 +/- 3.765% | 10 |

## Scenario S6

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 8.383 +/- 1.309 | 4.392 +/- 0.536 | 33.761 +/- 8.388% | 10 |
| GRU + IPPO | 7.787 +/- 1.539 | 4.096 +/- 0.562 | 31.986 +/- 7.620% | 10 |
| LSTM + IPPO | 9.148 +/- 1.262 | 4.636 +/- 0.541 | 36.942 +/- 7.285% | 10 |
| Transformer + MAPPO | 9.661 +/- 1.531 | 4.960 +/- 0.635 | 43.398 +/- 8.058% | 10 |
| GRU + MAPPO | 9.171 +/- 1.727 | 4.701 +/- 0.838 | 38.510 +/- 11.469% | 10 |
| LSTM + MAPPO | 9.084 +/- 1.513 | 4.630 +/- 0.595 | 39.384 +/- 9.293% | 10 |
| NoHistory + IPPO | 6.405 +/- 1.521 | 3.887 +/- 0.754 | 31.748 +/- 7.786% | 10 |
| NoHistory + MAPPO | 6.273 +/- 0.695 | 3.672 +/- 0.265 | 27.601 +/- 6.227% | 10 |

## Scenario S7

| Model | Dwell time (steps) | Interaction depth (techniques) | Asset protection | Seeds |
|---|---:|---:|---:|---:|
| Transformer + IPPO | 5.277 +/- 0.586 | 5.001 +/- 0.533 | 36.337 +/- 5.453% | 10 |
| GRU + IPPO | 5.240 +/- 0.302 | 4.960 +/- 0.267 | 39.779 +/- 4.765% | 10 |
| LSTM + IPPO | 5.094 +/- 0.481 | 4.833 +/- 0.422 | 38.447 +/- 6.808% | 10 |
| Transformer + MAPPO | 5.506 +/- 0.529 | 5.208 +/- 0.488 | 38.514 +/- 5.792% | 10 |
| GRU + MAPPO | 5.412 +/- 0.538 | 5.129 +/- 0.500 | 37.716 +/- 5.819% | 10 |
| LSTM + MAPPO | 5.649 +/- 0.445 | 5.332 +/- 0.387 | 40.598 +/- 4.477% | 10 |
| NoHistory + IPPO | 4.427 +/- 0.308 | 4.231 +/- 0.275 | 28.491 +/- 3.731% | 10 |
| NoHistory + MAPPO | 4.525 +/- 0.348 | 4.336 +/- 0.315 | 28.299 +/- 4.936% | 10 |

## Fixed comparisons

Two-sided paired seed tests; Holm correction across all 27 architecture/learner/metric comparisons.
For LOSO, each seed is averaged over its seven folds before testing.

| Contrast | Metric | Mean difference | Holm p |
|---|---|---:|---:|
| Transformer: MAPPO - IPPO | Dwell time (steps) | 0.5324 | 0.04268 |
| Transformer: MAPPO - IPPO | Interaction depth (techniques) | 0.2760 | 0.04122 |
| Transformer: MAPPO - IPPO | Asset protection | 0.0354 | 0.29566 |
| GRU: MAPPO - IPPO | Dwell time (steps) | 0.4468 | 0.32517 |
| GRU: MAPPO - IPPO | Interaction depth (techniques) | 0.2414 | 0.87414 |
| GRU: MAPPO - IPPO | Asset protection | -0.0054 | 1.00000 |
| LSTM: MAPPO - IPPO | Dwell time (steps) | 0.2708 | 1.00000 |
| LSTM: MAPPO - IPPO | Interaction depth (techniques) | 0.1750 | 1.00000 |
| LSTM: MAPPO - IPPO | Asset protection | 0.0200 | 1.00000 |
| IPPO: Transformer - GRU | Dwell time (steps) | 0.0816 | 1.00000 |
| IPPO: Transformer - GRU | Interaction depth (techniques) | 0.0512 | 1.00000 |
| IPPO: Transformer - GRU | Asset protection | -0.0214 | 1.00000 |
| IPPO: Transformer - LSTM | Dwell time (steps) | -0.1058 | 1.00000 |
| IPPO: Transformer - LSTM | Interaction depth (techniques) | -0.0214 | 1.00000 |
| IPPO: Transformer - LSTM | Asset protection | -0.0042 | 1.00000 |
| IPPO: GRU - LSTM | Dwell time (steps) | -0.1874 | 1.00000 |
| IPPO: GRU - LSTM | Interaction depth (techniques) | -0.0726 | 1.00000 |
| IPPO: GRU - LSTM | Asset protection | 0.0172 | 1.00000 |
| MAPPO: Transformer - GRU | Dwell time (steps) | 0.1672 | 1.00000 |
| MAPPO: Transformer - GRU | Interaction depth (techniques) | 0.0858 | 1.00000 |
| MAPPO: Transformer - GRU | Asset protection | 0.0194 | 1.00000 |
| MAPPO: Transformer - LSTM | Dwell time (steps) | 0.1558 | 1.00000 |
| MAPPO: Transformer - LSTM | Interaction depth (techniques) | 0.0796 | 1.00000 |
| MAPPO: Transformer - LSTM | Asset protection | 0.0112 | 1.00000 |
| MAPPO: GRU - LSTM | Dwell time (steps) | -0.0114 | 1.00000 |
| MAPPO: GRU - LSTM | Interaction depth (techniques) | -0.0062 | 1.00000 |
| MAPPO: GRU - LSTM | Asset protection | -0.0082 | 1.00000 |

## Limits

- Fixed five-zone engagement simulator, with four parameter-sharing agents.
- Synthetic Markov episodes estimated from verified CAM-LDS playbooks, not raw-log replay or live attacks.
- NoHistory still observes local telemetry and recent locally observed technique; it removes the shared history embedding.
- Reward optimizes dwell only. Depth and asset protection are evaluated outcomes, not guaranteed improvements.
- Architecture choices differ from historical v7; old checkpoints and reported scores are not reused.
- A small smoke run verifies execution only. It does not measure learning quality or support thesis claims.
