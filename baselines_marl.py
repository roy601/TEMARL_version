# -*- coding: utf-8 -*-
"""
TEMARL v7 — reference ladder (architecture-free)
================================================
Every learned arm is judged against these policies, and the validity gates
(`gates_marl.py`) are computed from them alone, before any learned arm exists.
None of them contains a neural network, so nothing here can favour one history
encoder over another.

INFORMATION LEVELS (what a planner is told about the attacker)
  clairvoyant : the next zone and the next two techniques, read from the
                pre-drawn episode (an upper-bound reference, not a policy an
                agent could run)
  transition  : the hidden script, hence its target zone and its transition
                row T_s[current technique]  -- "perfect intent + sequence"
  script      : the hidden script, hence its target zone and its average
                technique mix m_s, but NOT the current technique
                -- "perfect intent label, no sequence"
  bigram      : the current technique only; script pooled out
  nointent    : nothing about the script: the pooled technique mix and the
                prior over target zones
Every non-clairvoyant planner also knows the attacker's current zone and any
wired breadcrumb -- the union of what the four agents observe.

COORDINATION
  coordinated   : a central allocator. At most K decoys; a breadcrumb is only
                  laid together with a partner decoy in another zone.
  uncoordinated : every agent decides alone from the same beliefs, ignoring
                  the others and the capacity (variant `naive` may lay a
                  breadcrumb with no partner; variant `averse` never lays one).

Each planner has one parameter, a deployment threshold theta, chosen on a
separate SELECTION episode stream; it is then scored on the EVALUATION stream.
The best fixed joint action is chosen the same way from all 6^4 tuples.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np

import _frozen  # noqa: F401
from d3fend_payoff import N_ACTIONS, PAYOFF
from env_marl import (AGENT_ZONE, LURES, N_AGENTS, NULL, SCRIPT_IDS, ZONE_AGENT,
                      EngagementEnv, EnvConfig, EpisodeSource, run_episode)
from scripts_marl import scripts, step_marginals

THETAS = (0.0, 0.1, 0.2, 0.3, 0.4)
NONLURE = [d for d in range(1, N_ACTIONS) if d not in LURES]
DECOYS = list(range(1, N_ACTIONS))
METRICS = ("dwell", "depth", "protected", "exposed", "lure_chains",
           "dead_ends", "capacity_violations", "length")


# ── script summaries used by the planners ────────────────────────────────────

def script_mix(sid: str) -> np.ndarray:
    """Average technique mix of a script over its mean horizon."""
    sc = scripts()[sid]
    H = int(round(np.mean(sc["horizons"])))
    return step_marginals(sid, H).mean(0)


class Beliefs:
    """Per-pool precomputation: script mixes, pooled mix, target prior."""

    def __init__(self, pool: Sequence[str]):
        sc = scripts()
        self.pool = list(pool)
        # every script's own mix is known to the oracles that are TOLD the
        # script (also a held-out one); pooled quantities use the pool only
        self.mix = {s: script_mix(s) for s in sc}
        self.pooled = np.mean([self.mix[s] for s in self.pool], axis=0)
        self.target_prior = np.zeros(max(AGENT_ZONE) + 1)
        for s in self.pool:
            self.target_prior[sc[s]["target"]] += 1.0 / len(self.pool)
        # p(script | technique) for the pooled bigram belief
        M = np.stack([self.mix[s] for s in self.pool])          # (S, NT)
        self.post_by_tau = M / np.maximum(M.sum(0, keepdims=True), 1e-12)
        self.T = np.stack([sc[s]["T"] for s in self.pool])      # (S, NT, NT)
        self.init = np.stack([sc[s]["init"] for s in self.pool])


def next_zone_dist(env: EngagementEnv, target_dist: np.ndarray) -> np.ndarray:
    """Distribution of the attacker's next zone, over AGENT order, given the
    current zone, any wired breadcrumb, and a belief over its target zone."""
    q = np.zeros(N_AGENTS)
    if env.pending >= 0:
        q[ZONE_AGENT[env.pending]] = 1.0
        return q
    cfg = env.cfg
    for tgt, w in enumerate(target_dist):
        if w <= 0:
            continue
        if env.zone == tgt:
            q[ZONE_AGENT[tgt]] += w * cfg.p_stay
            others = [z for z in AGENT_ZONE if z != tgt]
            for z in others:
                q[ZONE_AGENT[z]] += w * (1 - cfg.p_stay) / len(others)
        else:
            q[ZONE_AGENT[tgt]] += w * cfg.p_goal
            others = [z for z in AGENT_ZONE if z != tgt]
            for z in others:
                q[ZONE_AGENT[z]] += w * (1 - cfg.p_goal) / len(others)
    return q


# ── policies ─────────────────────────────────────────────────────────────────

class Null:
    name = "null"

    def __call__(self, env):
        return [NULL] * N_AGENTS


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env):
        return list(self.rng.integers(0, N_ACTIONS, N_AGENTS))


class Reactive:
    """The natural uncoordinated defender, using ONLY its own observation: if
    the attacker is in my zone, deploy the decoy best aligned with the
    technique I just saw; otherwise deploy nothing. `naive` may lay a
    breadcrumb (with no guarantee of a partner); `averse` never does."""

    def __init__(self, lure_mode: str = "naive"):
        self.cand = DECOYS if lure_mode == "naive" else NONLURE
        self.name = "reactive/%s" % lure_mode

    def __call__(self, env):
        a = [NULL] * N_AGENTS
        if env.cur_tau >= 0 and env.zone in ZONE_AGENT:
            row = PAYOFF[env.cur_tau]
            a[ZONE_AGENT[env.zone]] = max(self.cand, key=lambda k: row[k])
        return a


class Fixed:
    def __init__(self, joint):
        self.joint = list(joint)
        self.name = "fixed%s" % (tuple(self.joint),)

    def __call__(self, env):
        return self.joint


class Planner:
    """Belief-based allocator; see the module docstring."""

    def __init__(self, info: str, coordinated: bool, beliefs: Beliefs,
                 theta: float = 0.0, lure_mode: str = "naive",
                 partner: bool = True, fill_all: bool = False):
        assert info in ("clairvoyant", "transition", "script", "bigram",
                        "nointent")
        self.info, self.coord, self.B = info, coordinated, beliefs
        self.theta, self.lure_mode = theta, lure_mode
        self.partner, self.fill_all = partner, fill_all
        tag = "coord" if coordinated else "uncoord-%s" % lure_mode
        if coordinated and not partner:
            tag += "-nopartner"
        if fill_all:
            tag += "-fillall"
        self.name = "%s/%s" % (info, tag)

    # beliefs about the next zone and the next two techniques
    def _beliefs(self, env):
        sc = scripts()[env.script]
        B, t = self.B, env.t
        if self.info == "clairvoyant":
            q = np.zeros(N_AGENTS)
            q[ZONE_AGENT[env.peek_next_zone()]] = 1.0
            p1 = np.zeros(PAYOFF.shape[0]); p1[env.spec.tau[t]] = 1.0
            p2 = None
            if t + 1 < env.spec.horizon:
                p2 = np.zeros(PAYOFF.shape[0]); p2[env.spec.tau[t + 1]] = 1.0
            return q, p1, p2
        if self.info in ("transition", "script"):
            tdist = np.zeros(len(B.target_prior)); tdist[sc["target"]] = 1.0
        else:
            tdist = B.target_prior
        q = next_zone_dist(env, tdist)
        if self.info == "transition":
            p1 = sc["init"] if env.cur_tau < 0 else sc["T"][env.cur_tau]
            p2 = p1 @ sc["T"]
        elif self.info == "script":
            p1 = p2 = B.mix[env.script] if env.script in B.mix else B.pooled
        elif self.info == "bigram":
            if env.cur_tau < 0:
                p1 = B.init.mean(0)
            else:
                w = B.post_by_tau[:, env.cur_tau]
                p1 = (w[:, None] * B.T[:, env.cur_tau, :]).sum(0)
            p2 = B.pooled
        else:
            p1 = p2 = B.pooled
        return q, p1, p2

    def __call__(self, env):
        q, p1, p2 = self._beliefs(env)
        ev1 = p1 @ PAYOFF                            # expected rho per type
        a = [NULL] * N_AGENTS
        if not self.coord:
            cand = DECOYS if self.lure_mode == "naive" else NONLURE
            d = max(cand, key=lambda k: ev1[k])
            for i in range(N_AGENTS):
                if q[i] > 0 and q[i] * ev1[d] >= self.theta:
                    a[i] = d
            return a
        order = sorted(range(N_AGENTS), key=lambda i: (-q[i], i))
        i1 = order[0]
        d1 = max(DECOYS, key=lambda k: ev1[k])
        if q[i1] * ev1[d1] < self.theta:
            return a
        target_agent = ZONE_AGENT[scripts()[env.script]["target"]] \
            if self.info in ("clairvoyant", "transition", "script") else None
        if d1 in LURES and not self.partner:
            a[i1] = d1
        elif d1 in LURES:
            # lay the breadcrumb only with a partner; prefer a partner outside
            # the target zone (it diverts the attacker away from real assets)
            ip = sorted((i for i in range(N_AGENTS) if i != i1),
                        key=lambda i: (i == target_agent, -q[i], i))[0]
            ev2 = (p2 if p2 is not None else p1) @ PAYOFF
            a[i1] = d1
            a[ip] = max(NONLURE, key=lambda k: ev2[k])
        else:
            a[i1] = d1
            if len(order) > 1:
                i2 = order[1]
                d2 = max(NONLURE, key=lambda k: ev1[k])
                if q[i2] > 0 and q[i2] * ev1[d2] >= self.theta:
                    a[i2] = d2
        if self.fill_all:
            for i in range(N_AGENTS):
                if a[i] == NULL:
                    a[i] = max(NONLURE, key=lambda k: ev1[k])
        return a


# ── evaluation helpers ───────────────────────────────────────────────────────

def evaluate(policy, specs, cfg: EnvConfig) -> List[Dict[str, float]]:
    env = EngagementEnv(cfg)
    return [run_episode(env, sp, policy) for sp in specs]


def summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {"n": len(rows)}
    for m in METRICS:
        x = np.array([r[m] for r in rows], float)
        out[m] = float(x.mean())
        out[m + "_ci"] = float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return out


def paired_diff(rows_a, rows_b, metric="dwell"):
    """Mean and 95% CI of a - b over the SAME episodes (common random numbers)."""
    d = np.array([a[metric] - b[metric] for a, b in zip(rows_a, rows_b)], float)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def tune_theta(make, sel_specs, cfg):
    """Pick theta on the SELECTION stream by mean dwell (ties -> smaller theta)."""
    best, best_v = None, -1.0
    for th in THETAS:
        v = np.mean([r["dwell"] for r in evaluate(make(th), sel_specs, cfg)])
        if v > best_v + 1e-12:
            best, best_v = th, v
    return best


def best_fixed(sel_specs, cfg):
    """Exhaustive search over all 6^4 constant joint actions on SELECTION."""
    scores = {}
    for joint in itertools.product(range(N_ACTIONS), repeat=N_AGENTS):
        scores[joint] = np.mean([r["dwell"] for r in evaluate(Fixed(joint), sel_specs, cfg)])
    best = max(scores, key=lambda j: (scores[j], -sum(x != 0 for x in j)))
    return list(best), scores


def ladder(cfg: EnvConfig, pool: Sequence[str] = SCRIPT_IDS, n_sel: int = 400,
           n_eval: int = 2000, n_fixed_sel: int = 150, seed: int = 0,
           eval_pool: Sequence[str] = None, verbose: bool = False) -> Dict:
    """Score every reference policy. Selection and evaluation streams are
    disjoint; evaluation episodes are shared by all policies (paired)."""
    B = Beliefs(pool)
    sel = EpisodeSource(10_000 + seed, pool).take(n_sel)
    sel_fixed = sel[:n_fixed_sel]
    ev = EpisodeSource(20_000 + seed, eval_pool or pool).take(n_eval)

    rows, meta = {}, {}
    rows["null"] = evaluate(Null(), ev, cfg)
    rows["random"] = evaluate(RandomPolicy(30_000 + seed), ev, cfg)
    bf, _ = best_fixed(sel_fixed, cfg)
    rows["best_fixed"] = evaluate(Fixed(bf), ev, cfg)
    meta["best_fixed_joint"] = bf
    for lm in ("naive", "averse"):
        rows["reactive/%s" % lm] = evaluate(Reactive(lm), ev, cfg)

    specs = []
    for info in ("clairvoyant", "transition", "script", "bigram", "nointent"):
        specs.append((info, True, "naive", True, False))
        specs.append((info, False, "naive", True, False))
        specs.append((info, False, "averse", True, False))
    specs.append(("clairvoyant", True, "naive", False, False))   # no partner
    specs.append(("clairvoyant", True, "naive", True, True))     # fill all
    for info, coord, lm, partner, fill in specs:
        def make(th, info=info, coord=coord, lm=lm, partner=partner, fill=fill):
            return Planner(info, coord, B, th, lm, partner, fill)
        th = tune_theta(make, sel, cfg)
        p = make(th)
        rows[p.name] = evaluate(p, ev, cfg)
        meta[p.name] = {"theta": th}
        if verbose:
            s = summarise(rows[p.name])
            print("    %-36s theta %.1f  dwell %6.3f  depth %6.3f  prot %.3f"
                  % (p.name, th, s["dwell"], s["depth"], s["protected"]))
    return {"rows": rows, "meta": meta,
            "summary": {k: summarise(v) for k, v in rows.items()}}


if __name__ == "__main__":
    import argparse
    import sys
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sel", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-fixed-sel", type=int, default=20)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    out = ladder(EnvConfig(), n_sel=args.n_sel, n_eval=args.n_eval,
                 n_fixed_sel=args.n_fixed_sel, verbose=True)
    print("  (smoke run of the ladder code only; %d eval episodes; %.0f s)"
          % (args.n_eval, time.time() - t0))
