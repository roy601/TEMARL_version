# Architecture mapping and audit decisions

## Review scope

Reviewed the repository layout and the progression from the fixed v2 system,
variable-topology v3 experiments, v5/v6 corrections, COMISET branch, to the
current engagement MARL branch. Examined the current runtime dependency chain,
environment dynamics, source grounding, pretraining, PPO rollout and value
updates, checkpoint selection, evaluation, and tests. Read the supplied progress
documents and relevant architecture sections of the three supplied papers.

This is an implementation audit of the selected runnable path. Historical
result archives and every binary checkpoint have not been independently
reproduced or scientifically validated. Passing software tests does not establish
that the simulator accurately represents live networks.

## Why the current engagement environment is retained

The latest progress documents describe four zone agents with capacity limits,
breadcrumbs and shared exposure. That corresponds to `temarl_marl`, not the
older entity-topology runner. Its engagement reward fixes the earlier mismatch
where successful capture ended an episode and reduced measured dwell.

The actor is parameter-shared among four agents and receives agent identity,
local telemetry and a shared SIEM history embedding. MAPPO uses a critic with
global state; IPPO uses a local critic. Both execute the same local actor
interface and use the same training and evaluation episode streams.

IPPO here means parameter-sharing independent PPO with local value estimates,
not four separately parameterized actor networks. NoHistory removes the history
embedding throughout both pretraining/policy learning and evaluation. A separate
test-time h=0 intervention is saved, but is not substituted for a trained control.

## Paper mapping

| Label | Source | Implementation and disclosed adaptations |
|---|---|---|
| Transformer | Manocchio et al., FlowTransformer, ESWA 241 (2024), 122564 | Two encoder layers, four attention heads, 64-dimensional embeddings, FFN 128, sinusoidal positions and last-real-token pooling. Uses ATT&CK tokens instead of flow records and a next-technique training head instead of NIDS labels. It is a paper-informed adaptation, not a reproduction of their full framework or reported experiment. |
| GRU | Cho et al., 1406.1078v3, section 2.3, equations 5-8 | Reset/update gates with candidate U(r*h), matching the paper's reset-before recurrence. Unlike the old built-in PyTorch GRU path, does not use reset-after. Affine biases, embedding, projection, normalization and modern AdamW/BPTT are task adaptations. |
| LSTM | Hochreiter and Schmidhuber (1997), sections 4 and 5 | One-cell blocks with input/output gates and a constant self-connection. No forget gate. Uses g(x)=4*sigmoid(x)-2 and h(c)=2*sigmoid(c)-1. The 1997 paper permits topology choices; this implementation uses recurrent cell outputs and stacked layers. Modern full BPTT replaces the paper's truncated update algorithm. |

All encoders have the same 64-dimensional interface, embedding vocabulary,
prediction head, 16-token rolling window, and pretraining recipe. Recurrent
width/layer counts are selected by parameter count only, within 10% of the
Transformer trunk, without consuming the model initialization RNG stream.

The full observed prefix is known when predicting its next token. A bidirectional
Transformer encoder within that prefix does not see the prediction target.
Padding is masked, pooling uses the true last token, and no future token enters
the decision history. No causal decoder or natural-language LLM is claimed.

The old Transformer-RoPE and modern forget-gate LSTM were intentional variants.
They were not silently treated as bugs. This new study uses different declared
architectures to align with the supplied papers; old model checkpoints are incompatible.

PyTorch documents its GRU reset-order difference explicitly:
https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

MAPPO design source: https://arxiv.org/abs/2103.01955
The retained project PPO is an adaptation, not a drop-in copy of that paper's
official benchmark implementation. PPO clipping, GAE, value normalization,
separate actor/critic optimizers and validation checkpoint selection are retained.

## Data provenance and intentional simplifications

1. The current verified JSON contains 36 runs and 1,347 labelled steps; older
   messages reporting 1,339 refer to an earlier file. This package pins the newer
   file by SHA-256 and tests its counts. No re-extraction is needed in the lab.
2. Transition statistics and run-length horizons use the verified sequences.
   Unknown techniques now raise an error rather than disappearing silently.
3. Entry/target zones and terminal-tactic declarations still use the old
   reconstruction. These are metadata, not the source of the technique chains.
   Therefore this is not a fully independently verified simulator.
4. Verified playbooks describe intended attack execution, not proof that each
   command succeeded or was observed in raw logs.
5. Episode sequences are sampled from a smoothed first-order Markov estimator.
   Smoothing can produce transitions and techniques absent from an individual
   scenario's playbooks. No exact transcript replay or long-range causal model
   is claimed. This is a limitation when interpreting history encoders.
6. S4's firewall target maps to its defended entry zone. S6's missing Exfiltration
   goal falls back to Collection. Those explicit pre-existing rules are retained.
7. Environment movement is an abstract zone transition model, not host-level
   routing through an independently validated firewall graph.
8. D3FEND payoff values and calibrated beta/p_goal/p_stay are retained. The
   numerical probabilities remain modeling assumptions, not measured live rates.

## Corrections and additions

| Finding | Action |
|---|---|
| Folder depended on parent repository paths | Bundle frozen dependencies and data; resolve them from this folder |
| Verified grounding absent from frozen integrity list | Add its SHA-256 to the mandatory import-time check |
| GRU+IPPO and LSTM+IPPO missing | Explicit full 3 x 2 design |
| NoHistory+IPPO missing | Train it separately, plus NoHistory+MAPPO for a cleaner intent ablation |
| Selecting best MAPPO from test results would bias ablation | Select using mean validation dwell only; stable encoder-order tie break |
| Capacity search instantiated many models and consumed RNG | Analytic recurrent counts and forked Transformer count probe |
| PAD/UNK logits included in training but removed in prediction scoring | Exclude those non-target logits in the new pretraining loss |
| Length arguments could leave non-PAD garbage visible to attention | Mask by true length, reject PAD within the real prefix, zero empty history |
| Stochastic evaluation created an unused seed generator | Use it explicitly for categorical sampling |
| Old runner could mix incomplete or stale output | Record complete expected cell plan, code/data fingerprint, config and checkpoint hash |
| Scientific metrics were scattered across output files | Generate six-arm, control, selected-ablation and per-scenario tables with CSV/JSON |

## Reporting rules

Dwell is the inherited training/selection objective. All requested metrics are
reported, including unfavorable ones. Asset protection is the binary avoidance
of the configured objective within the finite horizon; it is not the percentage
of hosts uncompromised. Interaction depth counts unique engaged technique IDs.

Use seed-level means for SD, confidence intervals and paired tests. Do not treat
thousands of correlated episodes as thousands of independent model runs. The
optional LOSO analysis averages seven folds within each seed before testing.
Twenty-seven fixed comparisons (3 encoder-pair contrasts per learner and 3
learner contrasts, each on 3 metrics) share one Holm correction family.
Selected-best ablations remain descriptive/exploratory; validation selection
avoids direct test selection but does not make the historical experiments new
confirmatory evidence. A non-significant contrast does not prove equivalence.

This specification is newly written after historical results were visible. It
must not be described as the old v7 pre-registration. Review/freeze it before
running the lab experiment. The inherited 2M-step budget was selected using
the old Transformer-RoPE learning gate, so convergence of all three new
architectures must still be assessed after training.

No architecture is guaranteed to win. Future changes to data, reward, budget,
or model equations require a new output folder and a disclosed new fingerprint.
