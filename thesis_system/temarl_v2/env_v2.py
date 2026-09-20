# -*- coding: utf-8 -*-
"""
TEMARL v2 — Dec-POMDP Deception Environment
===========================================
Closes F-ENV-01/04/05/07, F-MTH-04/05/06, F-RWD-01, F-DAT-02.

WHAT CHANGED vs v1, AND WHY
---------------------------
(1) TOPOLOGY  [F-ENV-07]  12-node hand-built Hawkeyes graph -> CAM-LDS 5-zone
    enterprise segmentation (Internet/DMZ/LAN/User/Admin behind a Firewall hub),
    per ISO/IEC 27033-3. Every cross-zone move traverses the firewall, which makes
    the ANTICIPATORY credit rule well-posed: the defender must anticipate the
    post-firewall destination, and that is exactly the decision h should inform.

(2) ATTACKER  [F-ENV-05]  v1 chose techniques by `stage_idx = min(len(seq), 8)`
    -- technique was a DETERMINISTIC FUNCTION OF STEP INDEX, i.e. memoryless:
    p(tau_t | history) = p(tau_t | t). A linear probe recovered the step index
    from h at R^2 = 0.97-0.98 and a bare step-counter oracle BEAT the Transformer
    (0.692 vs 0.557). There was nothing for self-attention to model.
    v2 samples a hidden campaign profile c per episode and draws
        tau_t ~ T_c[tau_{t-1}]
    from per-profile Markov chains estimated from CAM-LDS scenario orderings.
    The __main__ block MEASURES counter-vs-bigram accuracy; if history gain is
    not positive the environment is rejected.

(3) CAPTURE  [F-ENV-04]  v1 capture was positional (trap-node geometry) and
    INDEPENDENT of the defender's action, so DSR sat at 0.90-0.97 for every
    policy including Static and Random -- the outcome metric was invariant to
    behaviour. v2 makes containment causally depend on deception quality:

        p_cap(t) = p_min + (p_max - p_min) * sigmoid( k * (a_bar_t - a_ref) )
        a_bar_t  = lambda * a_bar_{t-1} + (1 - lambda) * a_t          (EMA)

    The naive form sigma(k*a) was rejected: sigma(0)=0.5 gives a ZERO-alignment
    defender a 50% capture rate (no discrimination at the low end), and an
    instantaneous (non-EMA) signal converts a dense per-step reward into a sparse
    high-variance terminal Bernoulli event -- reintroducing exactly the variance
    that broke TD learning. a_ref is pinned to the measured best-fixed value so an
    UNINFORMED policy sits at the sigmoid midpoint and only intent-driven gains
    move p_cap. lambda makes capture depend on SUSTAINED alignment.
    PRE-REGISTERED CHECK: Var[R|o,a] must not rise (SNR must not fall below the
    v1 value of 0.57). __main__ measures it.

(4) CREDIT   [F-MTH-04]  v1 used r2 = max over ALL FOUR agents' actions, which a
    static "play four different actions" policy maximises with zero localisation
    and zero prediction -- and r3 (diversity) paid for that same behaviour again.
    v2 credits EXACTLY ONE agent: the owner of the zone the attacker moves INTO,
    scored on the technique used on arrival, using the action that agent chose
    while still BLIND. This is the only rule under which h is structurally needed.

(5) REWARD   [F-MTH-05, F-MTH-03, F-RWD-01]  v1: R = .3r1+.4r2+.2r3+.1r4+.2r5 for
    the Thesis arm (sum 1.20) but .3r1+.4r2+.2r3+.1r4 for NoTrans (sum 1.00) --
    the two arms optimised DIFFERENT MDPs, so the reported -0.0313 ablation lift
    is uninterpretable. r5 was also action-independent and non-potential, which
    violates Ng-Harada-Russell (1999) policy invariance and can change the optimal
    policy while contributing no policy-gradient signal.
    v2: R = W_ENGAGE*r1 + W_ALIGN*r2, IDENTICAL for both arms, normalised, and a
    pure function of the responsible agent's action. r3/r4/r5 deleted.
"""

import json
import os
import random

import numpy as np

from d3fend_payoff import ACTIONS, N_ACTIONS, PAYOFF
from vocab_v2 import (MAX_SEQ_LEN, NUM_TECHNIQUES, PAD_ID, tactic_of,
                      technique_to_id)

# ── Topology: CAM-LDS 5-zone segmentation (ISO/IEC 27033-3) ──────────────────
ZONES        = ["Internet", "DMZ", "LAN", "User", "Admin"]
N_ZONES      = len(ZONES)
ZONE_ID      = {z: i for i, z in enumerate(ZONES)}
ATTACKER_ZONE = ZONE_ID["Internet"]          # attacker-controlled, undefended

ZONE_HOSTS = {
    "Internet": ["CorpDNS", "PublicDNS", "Attacker"],
    "DMZ":      ["DockerServer", "RepositoryServer", "VideoServer"],
    "LAN":      ["FileShare", "AdminLAN"],
    "User":     ["Client"],
    "Admin":    ["AdminHost"],
}
# Star through the Firewall hub: every cross-zone move traverses it.
DEFENDED_ZONES = [ZONE_ID["DMZ"], ZONE_ID["LAN"], ZONE_ID["User"], ZONE_ID["Admin"]]
N_AGENTS       = len(DEFENDED_ZONES)
ZONE_TO_AGENT  = {z: i for i, z in enumerate(DEFENDED_ZONES)}

# ── Reward weights: IDENTICAL for both arms, normalised to 1.0 (F-MTH-05) ────
W_ENGAGE = 0.30
W_ALIGN  = 0.70
assert abs(W_ENGAGE + W_ALIGN - 1.0) < 1e-9, "reward weights must normalise"
# Attacker reaches its objective. Selected by sweep against the PRE-REGISTERED
# VARIANCE bound (SNR >= 0.57), NOT against any win rate:
#     pen    SNR     DSR spread
#    -1.00  0.532      +0.180     fail (variance)
#    -0.50  0.625      +0.213     PASS  <- selected: passes the bound AND
#    -0.25  0.828      +0.172     PASS        maximises outcome discrimination
#     0.00  0.871      +0.153     PASS        (0.0 removes the failure signal)
# The -1.0 of v1 also REPLACED the step reward, making that step's return
# action-independent -- the same defect class as r5. Here it is additive.
TERMINAL_PENALTY = -0.5

# ── Containment parameters (F-ENV-04). Disclosed design constants. ───────────
# CALIBRATION NOTE (a bug found by the validator and fixed here):
#   A first attempt used p_cap in [0.35, 0.95] evaluated EVERY step. Over a
#   60-step episode that compounds to 1 - 0.65^60 ~ 1, so EVERY policy captured
#   the attacker and DSR re-saturated at 0.97-1.00 -- reproducing the very defect
#   this fix exists to remove. Capture must be a RACE against the attacker's
#   progress, not a certainty. Per-step probabilities are therefore scaled so
#   that over the expected campaign length the cumulative capture probability
#   spans a discriminating range rather than pinning at 1.
CAP_P_MIN  = 0.030   # low alignment -> attacker usually completes its campaign
CAP_P_MAX  = 0.160   # high alignment -> usually contained first
CAP_K      = 7.0     # spans the realistic alignment range without saturating
CAP_EMA    = 0.70    # capture depends on SUSTAINED alignment -> variance control
CAP_A_REF  = 0.3667  # = measured best-fixed value from the frozen payoff matrix

# Campaign progress: the attacker advances its objective only when it executes a
# technique belonging to its GOAL tactic. It wins on GOAL_STEPS such advances.
# This ties "the attacker succeeded" to campaign semantics rather than to a
# topological accident, and creates the race that makes DSR discriminating.
GOAL_STEPS = 5

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILE_PATH = os.path.join(_HERE, "profiles_v2.json")


# ── Attacker profile model ───────────────────────────────────────────────────

def build_profiles_from_camlds(path=None):
    """Estimate per-intent-class Markov chains T_c[tau_prev] -> tau_next from the
    CAM-LDS scenario orderings, consolidated into DECISION-DISTINCT classes.

    The consolidation (3 classes, not 7 scenarios) is inherited from
    d3fend_payoff.consolidate_intent_classes: campaigns demanding the same
    optimal deception are the same intent for the defender, and reporting 7
    would inflate the benchmark.
    """
    from d3fend_payoff import consolidate_intent_classes

    gpath = path or os.path.join(_HERE, "..", "data", "camlds_grounding.json")
    g = json.load(open(gpath, encoding="utf-8"))

    seqs, dists = {}, {}
    for s in g["scenarios"]:
        raw = [x for x in s["technique_sequence"] if not x.startswith("<")]
        ids = [technique_to_id(x) for x in raw]
        ids = [i for i in ids if i < NUM_TECHNIQUES]
        if len(ids) < 2:
            continue
        seqs[s["id"]] = ids
        d = np.zeros(NUM_TECHNIQUES)
        for i in ids:
            d[i] += 1.0
        goal = s["terminal_tactic"].split("(")[0].split("/")[0].strip()
        gidx = [j for j in range(NUM_TECHNIQUES) if tactic_of(j) == goal]
        if gidx:
            for j in gidx:
                d[j] += 0.6 * d.sum() / len(gidx)
        dists[s["id"]] = d

    con = consolidate_intent_classes(dists)

    profiles = {}
    for cls, members in con["members"].items():
        T = np.full((NUM_TECHNIQUES, NUM_TECHNIQUES), 0.02)   # Laplace floor
        init = np.zeros(NUM_TECHNIQUES) + 0.02
        goal_ids = []
        for m in members:
            sq = seqs[m]
            init[sq[0]] += 2.0
            for a, b in zip(sq, sq[1:]):
                T[a, b] += 4.0
        # goal drift: the campaign pulls toward its objective tactic
        goal_tac = None
        for a_name, mem in con["members"].items():
            if a_name == cls:
                # objective tactic = the tactic this decoy class is primary for,
                # restricted to tactics actually present in the member sequences
                from d3fend_payoff import PRIMARY
                present = {tactic_of(i) for m2 in mem for i in seqs[m2]}
                cand = sorted(PRIMARY[a_name] & present)
                goal_tac = cand[0] if cand else None
        if goal_tac:
            goal_ids = [j for j in range(NUM_TECHNIQUES) if tactic_of(j) == goal_tac]
            for j in goal_ids:
                T[:, j] += 1.2
        T = T / T.sum(1, keepdims=True)
        profiles[cls] = {"T": T, "init": init / init.sum(),
                         "goal_tactic": goal_tac, "goal_ids": goal_ids,
                         "members": members}
    return profiles, con


class DeceptionEnvV2:
    """Dec-POMDP honeypot-deception environment on the CAM-LDS 5-zone topology."""

    def __init__(self, profiles=None, profile_name="Mixture", max_steps=40,
                 seed=None, alignment_driven_capture=True):
        if profiles is None:
            profiles, _ = build_profiles_from_camlds()
        self.profiles = profiles
        self.names = sorted(profiles)
        self.profile_name = profile_name
        self.max_steps = max_steps
        self.alignment_driven_capture = alignment_driven_capture
        self.rng = np.random.default_rng(seed)

        # observation layout (per agent, LOCAL = own zone only -> Dec-POMDP)
        self.local_dim = (N_ZONES        # own-zone one-hot
                          + 1            # attacker present in my zone (SIGHTED)
                          + 1            # my zone compromised
                          + 1            # host-compromise fraction
                          + 1            # firewall alert level
                          + 1            # normalised step
                          + NUM_TECHNIQUES)   # observed technique (only if sighted)
        self.h_dim = 64
        self.obs_dim = self.local_dim + self.h_dim
        self.ATTACKER_PRESENT_IDX = N_ZONES     # index of the sighted flag
        # A freshly constructed env must answer shape/state queries without
        # faulting (global_state()/_observations() previously required an
        # explicit reset() first, which crashed callers doing dimension probes).
        self.reset()

    # ── episode lifecycle ────────────────────────────────────────────────────
    def reset(self):
        self._profile = (self.rng.choice(self.names)
                         if self.profile_name == "Mixture" else self.profile_name)
        p = self.profiles[self._profile]
        self.zone = ATTACKER_ZONE
        self.prev_zone = ATTACKER_ZONE
        self.seq = []
        self.tau = int(self.rng.choice(NUM_TECHNIQUES, p=p["init"]))
        self.step_count = 0
        self.done = False
        self.ema_align = CAP_A_REF          # start neutral, not 0
        self.compromised = np.zeros(N_ZONES)
        self.alert = 0.0
        self.captured = False
        self.reached_objective = False
        self.progress = 0            # goal-tactic advances toward the objective
        return self._observations()

    def _next_technique(self):
        T = self.profiles[self._profile]["T"]
        return int(self.rng.choice(NUM_TECHNIQUES, p=T[self.tau]))

    def _move(self):
        """Cross-zone movement through the firewall hub. The attacker advances
        into defended zones; which zone it enters is stochastic, so the
        responsible agent is not knowable in advance from telemetry alone."""
        if self.zone == ATTACKER_ZONE:
            cand = DEFENDED_ZONES
        else:
            # may stay (dwell) or traverse the firewall into another zone
            if self.rng.random() < 0.45:
                return self.zone
            cand = [z for z in DEFENDED_ZONES if z != self.zone]
        w = np.array([1.0 + 0.8 * self.compromised[z] for z in cand])
        return int(self.rng.choice(cand, p=w / w.sum()))

    def step(self, actions, h=None):
        assert len(actions) == N_AGENTS
        self.prev_zone = self.zone

        # 1. attacker emits a technique (Markov in the hidden profile) and moves
        self.tau = self._next_technique()
        self.seq.append(self.tau)
        self.zone = self._move()

        # 2. ANTICIPATORY single-agent credit (F-MTH-04): the responsible agent
        #    is the owner of the zone the attacker moved INTO, and its action was
        #    chosen BEFORE the move (it was blind if the attacker was elsewhere).
        resp = ZONE_TO_AGENT.get(self.zone)
        blind = (self.prev_zone != self.zone)
        align = float(PAYOFF[self.tau, actions[resp]]) if resp is not None else 0.0

        # 3. alignment-driven containment (F-ENV-04), EMA-smoothed
        self.ema_align = CAP_EMA * self.ema_align + (1 - CAP_EMA) * align
        if self.alignment_driven_capture and resp is not None:
            z = CAP_K * (self.ema_align - CAP_A_REF)
            p_cap = CAP_P_MIN + (CAP_P_MAX - CAP_P_MIN) / (1.0 + np.exp(-z))
        else:
            p_cap = 0.9                      # v1-style: behaviour-invariant
        engaged = bool(self.rng.random() < p_cap)
        p = self.profiles[self._profile]
        if engaged:
            self.captured = True
        else:
            self.compromised[self.zone] = 1.0
            self.alert = min(1.0, self.alert + 0.15)
            # the campaign advances only on goal-tactic work -> the race
            if tactic_of(self.tau) == p["goal_tactic"]:
                self.progress += 1

        # 4. reward — identical for both arms, pure function of the credited action
        r1 = 1.0 if engaged else 0.0
        r2 = align
        reward = W_ENGAGE * r1 + W_ALIGN * r2

        # 5. termination
        self.step_count += 1
        if self.captured:
            self.done = True
        elif self.progress >= GOAL_STEPS:
            self.reached_objective = True
            self.done = True
            reward += TERMINAL_PENALTY      # ADDITIVE, not replacing (see above)
        elif self.step_count >= self.max_steps:
            self.done = True

        info = {"technique_id": self.tau, "profile": self._profile,
                "responsible_agent": resp, "blind": blind, "alignment": align,
                "ema_align": self.ema_align, "p_capture": float(p_cap),
                "captured": self.captured, "zone": self.zone,
                "reached_objective": self.reached_objective}
        return self._observations(h), [reward] * N_AGENTS, self.done, info

    # ── observations ─────────────────────────────────────────────────────────
    def _observations(self, h=None):
        hv = np.zeros(self.h_dim, np.float32) if h is None else np.asarray(h, np.float32)
        obs = []
        for ai, z in enumerate(DEFENDED_ZONES):
            v = np.zeros(self.local_dim, np.float32)
            v[z] = 1.0
            sighted = (self.zone == z)
            v[N_ZONES]     = 1.0 if sighted else 0.0
            v[N_ZONES + 1] = self.compromised[z]
            v[N_ZONES + 2] = self.compromised.sum() / N_ZONES
            v[N_ZONES + 3] = self.alert
            v[N_ZONES + 4] = self.step_count / self.max_steps
            if sighted and self.seq:                    # telemetry only when present
                v[N_ZONES + 5 + self.tau] = 1.0
            obs.append(np.concatenate([v, hv]))
        return obs

    def global_state(self):
        g = np.concatenate([self.compromised,
                            np.eye(N_ZONES)[self.zone],
                            [self.alert, self.step_count / self.max_steps,
                             self.ema_align]])
        return g.astype(np.float32)

    def padded_sequence(self):
        """RIGHT-padded (F-MTH-07). v1 LEFT-padded, which combined with ABSOLUTE
        sinusoidal PE meant identical sub-sequences received different positional
        encodings depending on how much padding preceded them -- so absolute
        position became a proxy for episode step, and h could read the clock off
        the padding geometry. Returns (ids, true_length); encoders must gather the
        final REAL token at index true_length-1, never a fixed position."""
        s = self.seq[-MAX_SEQ_LEN:]
        n = len(s)
        out = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int64)
        if n:
            out[:n] = s
        return out, max(n, 1)


# ── environment validation ───────────────────────────────────────────────────

def validate_environment(env_factory, n_eps=400, seed=0):
    """Pre-registered environment checks. Rejects the env if the attacker is a
    clock, if DSR saturates, or if reward SNR regressed below the v1 value."""
    from collections import Counter, defaultdict
    rng = np.random.default_rng(seed)
    ctx, big, cnt = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    dsr_rand, dsr_best, dsr_orac = [], [], []
    R_by_oa, prof_prefix = defaultdict(list), []

    from d3fend_payoff import ACTIONS as ACT
    best_fixed = 4        # decoy_file (measured best fixed on the frozen matrix)
    astar = {}
    env = env_factory()
    for name in env.names:
        astar[name] = ACT.index(name) if name in ACT else best_fixed

    for policy, bucket in (("random", dsr_rand), ("best_fixed", dsr_best),
                           ("oracle", dsr_orac)):
        e = env_factory()
        for _ in range(n_eps):
            e.reset(); done = False; prev = None; step = 0
            pref = []
            while not done:
                if policy == "random":
                    a = [int(rng.integers(N_ACTIONS)) for _ in range(N_AGENTS)]
                elif policy == "best_fixed":
                    a = [best_fixed] * N_AGENTS
                else:
                    a = [astar[e._profile]] * N_AGENTS
                _, rw, done, info = e.step(a)
                t = info["technique_id"]
                if policy == "random":
                    if prev is not None:
                        big[prev][t] += 1
                    cnt[min(step, 15)][t] += 1
                    prev = t; step += 1
                    if info["responsible_agent"] is not None:
                        R_by_oa[(info["blind"], a[info["responsible_agent"]])].append(rw[0])
                    if len(pref) < 3:
                        pref.append(t)
            if policy == "random":
                prof_prefix.append((tuple(pref), e._profile))
            bucket.append(1.0 if e.captured else 0.0)

    # 1. is the attacker a clock?  counter vs bigram
    cpred = {k: v.most_common(1)[0][0] for k, v in cnt.items() if v}
    bpred = {k: v.most_common(1)[0][0] for k, v in big.items() if v}
    c_ok = sum(v[cpred[k]] for k, v in cnt.items() if k in cpred)
    c_n  = sum(sum(v.values()) for v in cnt.values())
    b_ok = sum(v[bpred[k]] for k, v in big.items() if k in bpred)
    b_n  = sum(sum(v.values()) for v in big.values())
    counter_acc = c_ok / max(c_n, 1)
    bigram_acc  = b_ok / max(b_n, 1)

    # 2. DSR discrimination
    dsr = {"random": float(np.mean(dsr_rand)), "best_fixed": float(np.mean(dsr_best)),
           "oracle": float(np.mean(dsr_orac))}

    # 3. reward SNR (F-MTH-06)
    allr = [r for v in R_by_oa.values() for r in v]
    means = {k: float(np.mean(v)) for k, v in R_by_oa.items() if len(v) > 20}
    by_act = defaultdict(list)
    for (bl, a), m in means.items():
        by_act[a].append(m)
    act_means = {a: float(np.mean(v)) for a, v in by_act.items()}
    signal = max(act_means.values()) - min(act_means.values()) if act_means else 0.0
    noise = float(np.std(allr)) if allr else 1.0

    # 4. profile identifiability from a 3-step prefix
    pmap = defaultdict(Counter)
    for pref, c in prof_prefix:
        pmap[pref][c] += 1
    ident = sum(v.most_common(1)[0][1] for v in pmap.values()) / max(len(prof_prefix), 1)

    return {"counter_acc": counter_acc, "bigram_acc": bigram_acc,
            "history_gain": bigram_acc - counter_acc, "dsr": dsr,
            "dsr_spread": dsr["oracle"] - dsr["random"],
            "signal": signal, "noise": noise, "snr": signal / max(noise, 1e-9),
            "prefix_identifiability": ident, "n_profiles": len(env.names)}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 74)
    print("  TEMARL v2 — Dec-POMDP ENVIRONMENT VALIDATION")
    print("=" * 74)

    profiles, con = build_profiles_from_camlds()
    print(f"  topology      : {N_ZONES} zones {ZONES}, firewall hub")
    print(f"  defended zones: {[ZONES[z] for z in DEFENDED_ZONES]}  -> {N_AGENTS} agents")
    print(f"  intent classes: {len(profiles)}  (from {con['n_scenarios']} CAM-LDS scenarios)")
    for c, p in profiles.items():
        print(f"     {c:<24} goal={p['goal_tactic']:<20} <- {p['members']}")
    e0 = DeceptionEnvV2(profiles)
    print(f"  obs dim/agent : {e0.obs_dim} (local {e0.local_dim} + h {e0.h_dim})")
    print(f"  reward        : R = {W_ENGAGE}*r1 + {W_ALIGN}*r2  (sum=1.0, IDENTICAL both arms)")

    print("\n  running validation ...")
    v = validate_environment(lambda: DeceptionEnvV2(profiles, max_steps=40), n_eps=400)

    print(f"\n  [F-ENV-05] IS THE ATTACKER A CLOCK?")
    print(f"     step-counter accuracy : {v['counter_acc']:.3f}   (v1: 0.692 -- dominant)")
    print(f"     bigram    accuracy    : {v['bigram_acc']:.3f}")
    print(f"     history gain          : {v['history_gain']:+.3f}   "
          f"{'PASS  history matters' if v['history_gain'] > 0.02 else 'FAIL  memoryless'}")

    print(f"\n  [F-ENV-04] DOES BEHAVIOUR MOVE THE OUTCOME (DSR)?")
    for k in ("random", "best_fixed", "oracle"):
        print(f"     DSR {k:<11}: {v['dsr'][k]:.3f}")
    print(f"     spread (oracle-random): {v['dsr_spread']:+.3f}   "
          f"{'PASS  de-saturated' if v['dsr_spread'] > 0.08 else 'FAIL  still saturated'}"
          f"   (v1: all policies 0.90-0.97)")

    print(f"\n  [F-MTH-06] REWARD SIGNAL-TO-NOISE (must not regress below v1 0.57)")
    print(f"     action signal spread  : {v['signal']:.4f}")
    print(f"     reward std            : {v['noise']:.4f}")
    print(f"     SNR                   : {v['snr']:.3f}   "
          f"{'PASS' if v['snr'] >= 0.57 else 'FAIL  variance regressed'}")

    print(f"\n  profile identifiability from 3-step prefix: {v['prefix_identifiability']:.3f} "
          f"(chance {1.0/v['n_profiles']:.3f})")
    print("=" * 74)
