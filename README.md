# Thesis version 2310

Copy this entire folder to the research lab PC. It includes all project code
and CAM-LDS grounding files needed by the experiment. No parent repository,
COMISET corpus, AttackBed checkout, or internet service is needed at runtime.
Python packages must be installed on the destination machine.

## What is included

| Encoder | IPPO | MAPPO |
|---|---|---|
| Transformer | Yes | Yes |
| GRU | Yes | Yes |
| LSTM | Yes | Yes |

Two separately trained controls, NoHistory + IPPO and NoHistory + MAPPO, are
also included. After all main cells complete, the report chooses the MAPPO
encoder with the highest mean validation dwell across seeds and shows:

1. Selected encoder + IPPO
2. NoHistory + IPPO
3. Selected encoder + MAPPO

The extra NoHistory + MAPPO row isolates the intent contribution under MAPPO.
NoHistory means no shared history embedding, not absence of local telemetry.

Every table reports **Dwell time (steps)**, **Interaction depth (techniques)**,
and **Asset protection**. There are overall and per-scenario tables, plus
machine-readable JSON/CSV, seed uncertainty, and fixed paired comparisons.

## Environment and dataset

This reuses the current `temarl_marl` engagement simulator: four defenders in
DMZ, LAN, User and Admin, plus the Internet zone. Its complexity comes from
capacity contention, cross-zone breadcrumb cooperation, partial observation,
shared exposure and hidden attacker objectives. It is a fixed five-zone
simulator, NOT the older variable-size v3 graph experiment.

The attacker chains are estimated from **36 verified playbooks, 7 scenarios,
1,347 labelled steps**. Episode techniques are sampled from those Markov
chains. They are not raw host logs or exact playbook execution. Scenario entry,
target and goal metadata still come from the bundled paper reconstruction;
the provenance distinction is documented in ARCHITECTURE_AND_AUDIT.md.

The calibrated environment settings and D3FEND payoff are preserved. No old
scores or trained models are presented as results for this new version.

## Metrics

- Dwell: total steps engaged with a decoy, not total episode length or seconds.
- Interaction depth: number of distinct ATT&CK technique IDs engaged by decoys.
- Asset protection: fraction of episodes ending before the attacker reaches
  its configured objective. It does not mean every host remained uncompromised.

Reward is dwell. Depth and protection are measured separately.

## Package map

| Location | Purpose |
|---|---|
| `encoders_marl.py` | Three paper-based history encoders |
| `env_marl.py`, `scripts_marl.py` | Coupled deception environment and CAM-LDS attacker |
| `mappo.py`, `pretrain_marl.py` | MAPPO/IPPO learning and frozen history pretraining |
| `experiment.py` | Six arms, controls, seeds, budgets and selection rule |
| `run_experiment.py` | Resumable CPU/CUDA runner |
| `report_results.py` | All tables and validation-selected ablation |
| `tests_marl.py`, `test_bundle.py` | Environment, equations and pipeline regression checks |
| `baselines_marl.py`, `gates_marl.py` | Retained architecture-free reference policies and gates |
| `thesis_system/` | Bundled frozen dependencies and both grounding files |
| `references/` | The three supplied architecture papers |
| `ARCHITECTURE_AND_AUDIT.md` | Intentional changes, limitations and source mapping |
| `RESULT_TABLES_TEMPLATE.md` | Expected tables, pending real training |
| `VALIDATION.md` | Local validation evidence and hardware limits |

## Environment requirements

Python 3.11 or newer, NumPy, SciPy and PyTorch. `requirements.txt` contains
package ranges. For the lab GPU, install a PyTorch CUDA build compatible with
the lab driver using the official selector: https://pytorch.org/get-started/locally/.
CUDA is used for neural operations; simulation and orchestration still use CPU.
Using `--device cuda` fails explicitly if CUDA is unavailable.

Detailed lab-running guidance will be discussed separately. Available entrypoints:

```text
python tests_marl.py
python test_bundle.py
python run_experiment.py --smoke --device cpu
python run_experiment.py --study main --device cuda
python run_experiment.py --study loso --device cuda
python report_results.py results/main
```

The main study contains 80 cells: 6 combinations plus 2 controls x 10 seeds.
The optional LOSO study trains on six scenarios and tests the seventh; it
contains 280 cells: 8 arms x 7 folds x 5 seeds. It measures unseen-scenario
transfer, not transfer to raw logs or unseen network sizes.

Default RL budget is 2,000,000 requested environment steps; the retained PPO
loop uses complete 4,096-step rollouts, so actual steps are 1,998,848 per cell.
The budget was inherited from v7 and is not newly validated for these exact
paper-based encoders. Check convergence before drawing architecture conclusions.

Runs resume at completed-cell boundaries. Checkpoints are evaluation-ready
actor/encoder snapshots, not mid-training optimizer-resume checkpoints.
An interrupted active cell restarts. A `.running` lock protects the output
folder; after a hard termination, check that the process ended before removing
that lock. Only one process may write to an output folder.

Reports in a `smoke_*` directory are execution checks, not scientific results.
The full experiment has not been run for this package.
