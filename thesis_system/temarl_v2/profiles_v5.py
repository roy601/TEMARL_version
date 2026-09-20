# -*- coding: utf-8 -*-
"""
TEMARL v5 — repaired attacker-model estimator
==============================================
ADDITIVE. `env_v2.py` and its `build_profiles_from_camlds()` are untouched and
remain the frozen control. `payoff_frozen.json` is untouched and is NOT refitted.
This module changes ONE thing in how the attacker's Markov chain is estimated.

WHAT IS BEING CORRECTED, AND WHY IT IS A CORRECTION
---------------------------------------------------
`env_v2.build_profiles_from_camlds()` estimates each campaign chain as

    T  = 0.02                      flat floor        1.68 mass/row
       + 4.0 per observed bigram   the real data     4.00 mass/row
       + 1.2 per goal technique, ADDED TO EVERY ROW  7.20-8.40 mass/row

The third term is a column-constant: `T[:, j] += 1.2` adds the same vector to
every row. A quantity identical in every row carries **zero** information about
the transition, by construction -- it is a prior on the marginal applied inside
a conditional. It also outweighs the real bigram evidence roughly 2:1, and it
concentrates each campaign's stationary distribution onto its goal tactic, whose
techniques share one optimal decoy.

Measured consequences in the shipped environment (`instrument_analysis.py`,
`baselines_entity.py`):

  * raw CAM-LDS sequences carry I(tau_next; tau) = 4.83 bits (85.4% predictable)
  * the shipped chains retain 0.627 bits within-campaign
  * a PERFECT next-technique model is worth +0.0001 payoff / +0.0005 DSR over a
    campaign-class label -- so the optimal action is a function of the class
    label alone and no history architecture can be distinguished on the task

Removing the column-constant is therefore an estimator correction, justified by
the estimator's own algebra and by the information it discards. It is NOT a
change made because a particular architecture lost, and it does not touch the
payoff matrix, the reward, the capture rule, the topology, or the vocabulary.

The "campaign pulls toward its objective" intuition the drift was meant to encode
is not discarded -- it is already carried by `init` (campaigns start where their
scenario starts) and by the GOAL_STEPS race in the environment. What is removed
is only its expression as a row-independent additive inside the transition model.

WHAT IS DELIBERATELY *NOT* CHANGED
-----------------------------------
`instrument_analysis.py` ran a clean 2x2x2 over {class granularity} x {estimator}
x {goal drift}. Two changes I had expected to help were measured to HURT:

    goal drift  on -> off     +0.0717   <- kept
    estimator   flat -> backoff  -0.0588   <- rejected
    classes     3 -> 7           -0.0305   <- rejected

Backoff smoothing and finer class granularity each make the CLASS LABEL more
informative, which shrinks the residual a sequence model could claim. So the
flat floor stays and the 3-class consolidation stays. Exactly one factor moves.
"""

from __future__ import annotations

import json
import os

import numpy as np

from d3fend_payoff import PRIMARY, consolidate_intent_classes
from env_v2 import NUM_TECHNIQUES, technique_to_id
from vocab_v2 import tactic_of

_HERE = os.path.dirname(os.path.abspath(__file__))
NT = NUM_TECHNIQUES

# Estimator constants, carried over UNCHANGED from env_v2 so that the drift is
# the only difference between v2 and v5 profiles.
FLOOR = 0.02
BIGRAM_WEIGHT = 4.0
INIT_WEIGHT = 2.0
GOAL_DRIFT = 0.0          # <-- the correction. env_v2 uses 1.2.

_CACHE = None


def build_profiles_v5(path=None, goal_drift: float = GOAL_DRIFT):
    """Identical to env_v2.build_profiles_from_camlds() except `goal_drift`.

    Passing goal_drift=1.2 reproduces the v2 profiles bit-for-bit, which is how
    the equivalence check in __main__ is done -- so the claim "exactly one factor
    changed" is verified, not asserted.
    """
    gpath = path or os.path.join(_HERE, "..", "data", "camlds_grounding.json")
    g = json.load(open(gpath, encoding="utf-8"))

    seqs, dists = {}, {}
    for s in g["scenarios"]:
        raw = [x for x in s["technique_sequence"] if not x.startswith("<")]
        ids = [technique_to_id(x) for x in raw]
        ids = [i for i in ids if i < NT]
        if len(ids) < 2:
            continue
        seqs[s["id"]] = ids
        d = np.zeros(NT)
        for i in ids:
            d[i] += 1.0
        goal = s["terminal_tactic"].split("(")[0].split("/")[0].strip()
        gidx = [j for j in range(NT) if tactic_of(j) == goal]
        if gidx:
            for j in gidx:
                d[j] += 0.6 * d.sum() / len(gidx)
        dists[s["id"]] = d

    con = consolidate_intent_classes(dists)

    profiles = {}
    for cls, members in con["members"].items():
        T = np.full((NT, NT), FLOOR)
        init = np.zeros(NT) + FLOOR
        for m in members:
            sq = seqs[m]
            init[sq[0]] += INIT_WEIGHT
            for a, b in zip(sq, sq[1:]):
                T[a, b] += BIGRAM_WEIGHT
        present = {tactic_of(i) for m2 in members for i in seqs[m2]}
        cand = sorted(PRIMARY[cls] & present) if cls in PRIMARY else []
        goal_tac = cand[0] if cand else None
        goal_ids = ([j for j in range(NT) if tactic_of(j) == goal_tac]
                    if goal_tac else [])
        if goal_drift:
            for j in goal_ids:
                T[:, j] += goal_drift
        T = T / T.sum(1, keepdims=True)
        profiles[cls] = {"T": T, "init": init / init.sum(),
                         "goal_tactic": goal_tac, "goal_ids": goal_ids,
                         "members": members}
    return profiles, con


def profiles_v5():
    """Module-level cache; rebuilding per episode dominated v3 runtime."""
    global _CACHE
    if _CACHE is None:
        _CACHE = build_profiles_v5()[0]
    return _CACHE


if __name__ == "__main__":
    import sys
    from d3fend_payoff import PAYOFF
    from env_v2 import build_profiles_from_camlds
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 88)
    print("  TEMARL v5 — REPAIRED ESTIMATOR")
    print("=" * 88)

    # 1. exactly-one-factor check
    v2p, _ = build_profiles_from_camlds()
    rep, _ = build_profiles_v5(goal_drift=1.2)
    worst = max(float(np.abs(v2p[c]["T"] - rep[c]["T"]).max()) for c in v2p)
    print("  [1] setting goal_drift=1.2 reproduces env_v2 profiles:")
    print("      max |T_v2 - T_v5(drift=1.2)| = %.3e   %s"
          % (worst, "PASS" if worst < 1e-12 else "FAIL"))

    v5p = profiles_v5()

    # 2. information retained
    def cond_mi(profs, seed=0):
        out = []
        for c, p in profs.items():
            T = np.asarray(p["T"])
            rng = np.random.default_rng(seed)
            occ = np.zeros(NT)
            tau = int(rng.choice(NT, p=p["init"]))
            for i in range(120000):
                tau = int(rng.choice(NT, p=T[tau])); occ[tau] += 1
                if (i + 1) % 40 == 0:
                    tau = int(rng.choice(NT, p=p["init"]))
            occ /= occ.sum()
            def H(x):
                x = np.asarray(x, float); x = x[x > 0]
                return float(-(x * np.log2(x)).sum())
            out.append(H(occ @ T) - sum(occ[t] * H(T[t]) for t in range(NT)))
        return float(np.mean(out))

    print("\n  [2] within-campaign sequential information I(tau'; tau | c)")
    print("      env_v2 : %.4f bits" % cond_mi(v2p))
    print("      v5     : %.4f bits" % cond_mi(v5p))
    print("      (raw CAM-LDS sequences carry 4.83 bits)")

    # 3. the instrument criterion
    def seq_value(profs, seed=0, n=80000):
        rng = np.random.default_rng(seed)
        occ, pairs = {}, {}
        for c, p in profs.items():
            s, tau = [], int(rng.choice(NT, p=p["init"]))
            for i in range(n):
                nxt = int(rng.choice(NT, p=p["T"][tau]))
                s.append((tau, nxt)); tau = nxt
                if (i + 1) % 40 == 0:
                    tau = int(rng.choice(NT, p=p["init"]))
            pairs[c] = np.array(s)
            occ[c] = np.bincount(pairs[c][:, 1], minlength=NT) / len(s)
        po = float(np.mean([(occ[c] @ PAYOFF).max() for c in profs]))
        tot, m = 0.0, 0
        for c, p in profs.items():
            a = (np.asarray(p["T"]) @ PAYOFF).argmax(1)
            pr = pairs[c]
            tot += float(PAYOFF[pr[:, 1], a[pr[:, 0]]].sum()); m += len(pr)
        bf = float(np.mean([occ[c] @ PAYOFF for c in profs], axis=0).max())
        nd = float(np.mean([len(set((np.asarray(p["T"]) @ PAYOFF).argmax(1)))
                            for p in profs.values()]))
        return bf, po, tot / m, tot / m - po, nd

    print("\n  [3] SEQUENCE VALUE — what a perfect next-technique model buys over")
    print("      a perfect campaign label. Must be non-zero for ANY history")
    print("      architecture to be distinguishable on this task.")
    print("      %-10s %9s %9s %9s %11s %9s"
          % ("design", "best-fix", "class-orc", "seq-orc", "SEQ VALUE", "a*/class"))
    for nm, pp in (("env_v2", v2p), ("v5", v5p)):
        bf, po, so, sv, nd = seq_value(pp)
        print("      %-10s %9.4f %9.4f %9.4f %+11.4f %9.2f"
              % (nm, bf, po, so, sv, nd))
    print("=" * 88)
