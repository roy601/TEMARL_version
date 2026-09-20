# -*- coding: utf-8 -*-
"""
TEMARL v7 — the attacker, grounded per real CAM-LDS attack script
=================================================================
v5 pooled the seven CAM-LDS scenarios into three decision-distinct intent
classes. v7 needs the scripts individually, for two reasons fixed by the
proposal:

  * GENERALISATION is defined as "performance against a novel, unseen attack
    script after training". That requires holding out one script at a time
    (leave-one-script-out), so each script needs its own model.
  * Each script has its own real GEOGRAPHY -- where it enters the network and
    which zone holds its target (`entry_zone`, `target_zone` in
    `camlds_grounding.json`). A pooled class would mix geographies that never
    co-occur in the data.

THE ESTIMATOR IS NOT CHANGED
----------------------------
Each chain is built by exactly the v5 estimator (`profiles_v5`: FLOOR 0.02,
BIGRAM_WEIGHT 4.0, INIT_WEIGHT 2.0, goal drift 0). This is verified, not
asserted: `estimate_chain()` applied to the v5 grouping of the v5 input
reproduces `profiles_v5()` bit-for-bit (`equivalence_check()`).

THE INPUT IS THE SOURCE-VERIFIED RUNS (a disclosed change from v5)
------------------------------------------------------------------
v5 estimated from the paper-based reconstruction (`camlds_grounding.json`,
one sequence per scenario). Its preprocessing drops the `<variant>`
placeholders, and those placeholders are exactly where some scripts' objectives
live: S1's goal tactic (Persistence) occurs NOWHERE in its reconstructed
sequence, so an S1 attacker could only ever reach its objective through the
estimator's 0.02 floor. v7 therefore estimates each chain from the 36
source-verified AttackBed runs (`camlds_grounding_verified.json`, whose own
provenance states it supersedes the reconstruction): every run of a script is
one member sequence. The estimator itself is untouched.

PRE-DECLARED RULES (fixed on 2026-09-18, before any v7 result existed)
----------------------------------------------------------------------
  target zone  : if a script's target is not a defended zone (S4 targets the
                 Firewall hub itself), its goal zone is its ENTRY zone -- it
                 attacks the firewall from where it stands.
  goal tactic  : the script's declared `terminal_tactic`; a parenthesised
                 tactic is used if present (S4: "Network Boundary Bridging
                 (Defense Evasion)"), else the first of a list (S7:
                 "Credential Access / Exfiltration").
  goal fallback: if the declared goal tactic occurs in NONE of the script's
                 verified runs and it is Exfiltration, the goal is Collection,
                 its ATT&CK prerequisite (data must be collected before it can
                 be exfiltrated). This fires for S6 only: its objective is
                 "KEYLOG / COLLECT and EXFILTRATE credentials", its runs are
                 dominated by keylogging and clipboard collection (T1056,
                 T1115), and exfiltration is never a separately executed step.
                 Any other unobservable goal is a hard error, not a guess.
  horizon      : an episode lasts as long as a real run of its script -- the
                 horizon is drawn uniformly from that script's verified run
                 lengths. A playbook has a fixed number of actions; deception
                 changes what they hit, not how many there are.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

import _frozen  # noqa: F401  (verifies frozen inputs, sets sys.path)
from _frozen import DATA_DIR
from env_v2 import DEFENDED_ZONES, ZONES, technique_to_id
from profiles_v5 import BIGRAM_WEIGHT, FLOOR, INIT_WEIGHT
from vocab_v2 import NUM_TECHNIQUES, TACTIC_ORDER, tactic_of

NT = NUM_TECHNIQUES
ZONE_ID = {z: i for i, z in enumerate(ZONES)}
GROUNDING_PATH = os.path.join(DATA_DIR, "camlds_grounding.json")
VERIFIED_PATH = os.path.join(DATA_DIR, "camlds_grounding_verified.json")

_CACHE = None


def _grounding():
    with open(GROUNDING_PATH, encoding="utf-8") as f:
        return json.load(f)


def _verified():
    with open(VERIFIED_PATH, encoding="utf-8") as f:
        return json.load(f)


def _to_ids(raw_seq) -> List[int]:
    """profiles_v5 preprocessing: drop placeholders, map ids, drop out-of-vocab."""
    raw = [x for x in raw_seq if not x.startswith("<")]
    ids = [technique_to_id(x) for x in raw]
    return [i for i in ids if i < NT]


def reconstruction_sequences() -> Dict[str, List[int]]:
    """The v5 input (one reconstructed sequence per scenario). Used only for
    the estimator equivalence check."""
    out = {}
    for s in _grounding()["scenarios"]:
        ids = _to_ids(s["technique_sequence"])
        if len(ids) >= 2:
            out[s["id"]] = ids
    return out


def verified_runs() -> Dict[str, List[List[int]]]:
    """v7 input: every source-verified run, grouped by script, in run-key order."""
    out: Dict[str, List[List[int]]] = {}
    runs = _verified()["runs"]
    for key in sorted(runs):
        r = runs[key]
        raw = r["sequence"]
        ids = [technique_to_id(x) for x in raw]
        if any(i >= NT for i in ids):
            raise ValueError(f"Verified run {key} contains an unknown technique")
        if len(ids) >= 2:
            out.setdefault(r["scenario"], []).append(ids)
    return out


def estimate_chain(member_seqs: List[List[int]]):
    """The v5 estimator, verbatim, for an arbitrary group of sequences.

    Accumulation order matches profiles_v5.build_profiles_v5 exactly (members
    in order; init then bigrams per member; one row normalisation at the end),
    which is what makes the equivalence check bit-exact.
    """
    T = np.full((NT, NT), FLOOR)
    init = np.zeros(NT) + FLOOR
    for sq in member_seqs:
        init[sq[0]] += INIT_WEIGHT
        for a, b in zip(sq, sq[1:]):
            T[a, b] += BIGRAM_WEIGHT
    T = T / T.sum(1, keepdims=True)
    return T, init / init.sum()


def _declared_goal(terminal: str) -> str:
    s = terminal.strip()
    if "(" in s and ")" in s:
        s = s[s.index("(") + 1: s.index(")")]
    s = s.split("/")[0].strip()
    if s not in TACTIC_ORDER:
        raise ValueError(f"goal tactic '{s}' (from '{terminal}') is not an "
                         f"ATT&CK tactic in the vocabulary")
    return s


def _resolve_goal(sid: str, declared: str, runs: List[List[int]]):
    observed = {tactic_of(i) for r in runs for i in r}
    if declared in observed:
        return declared, "declared"
    if declared == "Exfiltration" and "Collection" in observed:
        return "Collection", "fallback: Exfiltration unobserved -> Collection"
    raise ValueError(f"{sid}: declared goal tactic '{declared}' never occurs in "
                     f"its verified runs and no pre-declared fallback applies")


def _target_zone(entry: int, target_name: str) -> int:
    z = ZONE_ID.get(target_name)
    if z is not None and z in DEFENDED_ZONES:
        return z
    if entry not in DEFENDED_ZONES:
        raise ValueError(f"target '{target_name}' is not a defended zone and "
                         f"the entry zone is not defended either")
    return entry


def build_scripts() -> Dict[str, dict]:
    """One attacker model per CAM-LDS script."""
    runs = verified_runs()
    out = {}
    for s in _grounding()["scenarios"]:
        sid = s["id"]
        if sid not in runs:
            raise ValueError(f"{sid} has no verified runs")
        rs = runs[sid]
        T, init = estimate_chain(rs)
        entry = ZONE_ID[s["entry_zone"]]
        target = _target_zone(entry, s["target_zone"])
        declared = _declared_goal(s["terminal_tactic"])
        goal, goal_rule = _resolve_goal(sid, declared, rs)
        goal_ids = [j for j in range(NT) if tactic_of(j) == goal]
        out[sid] = {
            "id": sid, "name": s["name"],
            "T": T, "init": init,
            "cumT": np.cumsum(T, axis=1), "cuminit": np.cumsum(init),
            "entry": entry, "target": target, "target_raw": s["target_zone"],
            "goal_tactic": goal, "goal_declared": declared,
            "goal_rule": goal_rule, "goal_ids": goal_ids,
            "goal_mask": np.isin(np.arange(NT), goal_ids),
            "horizons": sorted(len(r) for r in rs),
            "n_runs": len(rs),
            "goal_steps_per_run": float(np.mean(
                [sum(tactic_of(i) == goal for i in r) for r in rs])),
        }
    return out


def scripts() -> Dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = build_scripts()
    return _CACHE


def script_ids() -> List[str]:
    return sorted(scripts())


def max_horizon() -> int:
    return max(max(sc["horizons"]) for sc in scripts().values())


def loso_folds() -> List[dict]:
    """Leave-one-script-out folds: train on six scripts, test on the seventh."""
    ids = script_ids()
    return [{"held_out": h, "train": [s for s in ids if s != h]} for h in ids]


def step_marginals(sid: str, n_steps: int) -> np.ndarray:
    """Exact per-step distribution of the executed technique, (n_steps, NT).
    Executed technique 1 ~ init, technique t+1 ~ T[technique t]."""
    sc = scripts()[sid]
    out = np.zeros((n_steps, NT))
    p = sc["init"].copy()
    for t in range(n_steps):
        out[t] = p
        p = p @ sc["T"]
    return out


def equivalence_check() -> dict:
    """The v5 grouping of the v5 input must reproduce profiles_v5() exactly."""
    from profiles_v5 import build_profiles_v5
    v5p, con = build_profiles_v5()
    seqs = reconstruction_sequences()
    worst_T, worst_init, exact = 0.0, 0.0, True
    for cls, members in con["members"].items():
        T, init = estimate_chain([seqs[m] for m in members])
        exact &= bool(np.array_equal(T, v5p[cls]["T"])
                      and np.array_equal(init, v5p[cls]["init"]))
        worst_T = max(worst_T, float(np.abs(T - v5p[cls]["T"]).max()))
        worst_init = max(worst_init,
                         float(np.abs(init - v5p[cls]["init"]).max()))
    return {"bit_exact": exact, "max_abs_T": worst_T,
            "max_abs_init": worst_init, "groups": con["members"]}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 100)
    print("  TEMARL v7 — per-script attacker models (source-verified runs)")
    print("=" * 100)
    eq = equivalence_check()
    print("  estimator identity: v5 grouping of the v5 input reproduces "
          "profiles_v5(): %s  (max |dT| %.1e)"
          % ("PASS" if eq["bit_exact"] else "FAIL", eq["max_abs_T"]))
    print("\n  %-3s %-21s %-8s %-8s %-18s %4s %-12s %s"
          % ("id", "name", "entry", "target", "goal tactic", "runs",
             "horizon", "goal steps/run"))
    for sid in script_ids():
        sc = scripts()[sid]
        tgt = ZONES[sc["target"]] + ("*" if sc["target_raw"] != ZONES[sc["target"]]
                                     else "")
        goal = sc["goal_tactic"] + ("+" if sc["goal_rule"] != "declared" else "")
        hz = sc["horizons"]
        print("  %-3s %-21s %-8s %-8s %-18s %4d %3d..%-7d %.1f"
              % (sid, sc["name"][:21], ZONES[sc["entry"]], tgt, goal,
                 sc["n_runs"], hz[0], hz[-1], sc["goal_steps_per_run"]))
    print("  * non-defended target -> entry zone;  + goal fallback "
          "(Exfiltration unobserved -> Collection)")
    print("  max horizon: %d steps" % max_horizon())
    print("=" * 100)
