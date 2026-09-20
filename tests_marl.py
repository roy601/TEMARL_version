# -*- coding: utf-8 -*-
"""
TEMARL v7 — unit tests for every environment mechanic
=====================================================
Each mechanic is exercised with a hand-built EpisodeSpec whose uniforms are
chosen to force one branch, so a test fails only if the rule itself is wrong.

    python tests_marl.py
"""

from __future__ import annotations

import sys

import numpy as np

import _frozen
from d3fend_payoff import PAYOFF
from env_marl import (AGENT_ZONE, GLOBAL_DIM, LOCAL_DIM, LURES, N_AGENTS, N_U,
                      O_SIGHTED, O_TECH, O_WIRED, U_BURN, U_DEAD, U_ENGAGE,
                      U_LATERAL, U_MOVE, U_WIRE, ZONE_AGENT, EngagementEnv,
                      EnvConfig, EpisodeSource, EpisodeSpec, run_episode)
from env_v2 import ZONES
from scripts_marl import equivalence_check, scripts
from vocab_v2 import TACTIC_TO_IDS

Z = {z: i for i, z in enumerate(ZONES)}
AG = {z: ZONE_AGENT[Z[z]] for z in ("DMZ", "LAN", "User", "Admin")}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  (%s)" % detail) if detail else ""))


def tech(tactic):
    return int(TACTIC_TO_IDS[tactic][0])


def spec(script, taus, **u_over):
    """A spec with every uniform 0.5 unless overridden per (row, step)."""
    h = len(taus)
    u = np.full((N_U, h), 0.5)
    for key, vals in u_over.items():
        row = {"move": U_MOVE, "lat": U_LATERAL, "engage": U_ENGAGE,
               "burn": U_BURN, "dead": U_DEAD, "wire": U_WIRE}[key]
        for t, v in vals.items():
            u[row, t] = v
    return EpisodeSpec(script, h, np.asarray(taus, np.int64), u)


def joint(**by_zone):
    a = [0] * N_AGENTS
    for z, d in by_zone.items():
        a[AG[z]] = d
    return a


CFG = EnvConfig(beta=0.3, p_goal=0.6, p_stay=0.7)


def test_scripts():
    eq = equivalence_check()
    check("chain estimator identical to profiles_v5 (bit-exact)", eq["bit_exact"])
    sc = scripts()
    check("7 scripts, each goal tactic observable",
          len(sc) == 7 and all(len(s["goal_ids"]) > 0 for s in sc.values()))
    check("S6 goal fallback is the only non-declared goal",
          [k for k, s in sc.items() if s["goal_rule"] != "declared"] == ["S6"])


def test_capacity():
    disc = tech("Discovery")                      # D3-DNR (a1) is PRIMARY: 0.9
    rho = (2 / 3) * PAYOFF[disc, 1]               # 3 active > K=2 -> phi = 2/3
    for u_e, want in ((rho - 0.01, True), (rho + 0.01, False)):
        env = EngagementEnv(CFG).reset(
            spec("S2", [disc, disc], move={0: 0.0}, engage={0: u_e},
                 burn={0: 0.99}))
        _, _, info = env.step(joint(DMZ=1, LAN=2, User=5))
        check("capacity: 3 decoys -> phi 2/3, engage iff u < %.3f (u=%.3f)"
              % (rho, u_e), info["engaged"] is want and abs(info["phi"] - 2 / 3) < 1e-12)
    env = EngagementEnv(CFG).reset(spec("S2", [disc], move={0: 0.0}, engage={0: 0.0}))
    _, _, info = env.step(joint(DMZ=1, Admin=2))
    check("capacity: 2 decoys -> phi 1", info["phi"] == 1.0)


def test_breadcrumb_chain():
    cred = tech("Credential Access")              # D3-DUC (a3) PRIMARY
    disc = tech("Discovery")
    env = EngagementEnv(CFG).reset(
        spec("S2", [cred, disc, disc], move={0: 0.0, 1: 0.0},
             engage={0: 0.0, 1: 0.0}, burn={0: 0.99, 1: 0.99}, wire={0: 0.0}))
    _, _, info = env.step(joint(DMZ=LURES[0], Admin=1))
    check("breadcrumb engaged and wired to the only partner (Admin)",
          info["engaged"] and info["wired"] == Z["Admin"] and env.pending == Z["Admin"])
    o = env.local_obs()
    check("the partner (and only the partner) sees the wired flag",
          o[AG["Admin"], O_WIRED] == 1.0
          and o[[AG["DMZ"], AG["LAN"], AG["User"]], O_WIRED].sum() == 0.0)
    # u_move = 0 would keep the attacker in its target (DMZ); the breadcrumb wins
    _, _, info = env.step(joint(Admin=1))
    check("attacker follows the breadcrumb, overriding its own movement",
          info["zone"] == Z["Admin"] and info["followed"])
    check("lure -> partner engagement counted as a completed chain",
          info["engaged"] and env.st["lure_chains"] == 1)


def test_dead_end():
    cred = tech("Credential Access")
    disc = tech("Discovery")
    env = EngagementEnv(CFG).reset(
        spec("S2", [cred, disc], move={0: 0.0, 1: 0.0}, engage={0: 0.0, 1: 0.0},
             burn={0: 0.99, 1: 0.0}, dead={0: 0.0}))
    _, _, info = env.step(joint(DMZ=LURES[0]))
    check("lure without a partner is a dead end",
          env.st["dead_ends"] == 1 and info["wired"] == -1)
    check("dead end exposes the deception when u_dead < beta", info["exposed"])
    _, _, info = env.step(joint(DMZ=1))
    check("after exposure a perfectly aligned decoy is ignored",
          not info["engaged"] and env.burned)


def test_exposure_hazard():
    disc = tech("Discovery")
    rho = PAYOFF[disc, 1]
    haz = CFG.beta * (1 - rho)
    for u_b, want in ((haz - 1e-3, True), (haz + 1e-3, False)):
        env = EngagementEnv(CFG).reset(
            spec("S2", [disc], move={0: 0.0}, engage={0: 0.0}, burn={0: u_b}))
        _, _, info = env.step(joint(DMZ=1))
        check("exposure iff u_burn < beta*(1-rho) = %.4f (u=%.4f)" % (haz, u_b),
              info["exposed"] is want)


def test_progress_rules():
    imp = tech("Impact")                          # S3 goal tactic, target LAN
    disc = tech("Discovery")
    # entry Internet -> target LAN (u_move=0 < p_goal), no decoy: objective
    env = EngagementEnv(CFG).reset(spec("S3", [imp, disc], move={0: 0.0}))
    _, done, info = env.step(joint())
    check("goal technique on a real host in the target zone ends the episode",
          done and env.objective and info["zone"] == Z["LAN"])
    # same, but an aligned decoy engages it: no progress
    env = EngagementEnv(CFG).reset(
        spec("S3", [imp, disc], move={0: 0.0}, engage={0: 0.0}, burn={0: 0.99},
             dead={0: 0.99}))
    _, done, info = env.step(joint(LAN=LURES[1]))
    check("an engaged goal technique touches no real asset", info["engaged"]
          and not env.objective and not done)
    # goal technique in a NON-target zone: no progress
    env = EngagementEnv(CFG).reset(
        spec("S3", [imp, disc], move={0: 0.99}, lat={0: 0.0}))
    _, done, info = env.step(joint())
    check("goal technique outside the target zone is not the objective",
          info["zone"] != Z["LAN"] and not env.objective and not done)


def test_horizon_and_reward():
    disc = tech("Discovery")                      # never S2's goal (Exfiltration)
    env = EngagementEnv(CFG).reset(spec("S2", [disc] * 7, burn={t: 0.99 for t in range(7)}))
    total, n = 0.0, 0
    while not env.done:
        r, _, _ = env.step(joint(DMZ=1))
        total += r; n += 1
    st = env.episode_stats()
    check("episode ends exactly at the horizon", n == 7 and st["length"] == 7)
    check("return equals dwell", total == st["dwell"])
    check("depth counts UNIQUE techniques", st["depth"] == 1.0 and st["dwell"] >= 1)


def test_locality():
    rng = np.random.default_rng(0)
    src = EpisodeSource(123)
    env = EngagementEnv(CFG)
    bad = 0
    for _ in range(60):
        env.reset(src.next())
        while not env.done:
            env.step(list(rng.integers(0, 6, N_AGENTS)))
            if env.done:
                break
            for i, z in enumerate(AGENT_ZONE):
                base = env.local_obs()[i].copy()
                saved = (env.decoys.copy(), env.compromised.copy(), env.zone)
                others = [k for k in range(N_AGENTS) if k != i]
                # change other agents' decoy TYPES (same active count)
                for k in others:
                    if env.decoys[k] != 0:
                        env.decoys[k] = 1 + (env.decoys[k] % 5)
                env.compromised[[AGENT_ZONE[k] for k in others]] = 1.0
                if env.zone != z and env.pending < 0:
                    env.zone = AGENT_ZONE[others[int(rng.integers(3))]]
                if not np.array_equal(base, env.local_obs()[i]):
                    bad += 1
                env.decoys, env.compromised, env.zone = saved
    check("an agent's observation is invariant to other zones' private state",
          bad == 0, "%d violations" % bad)
    o = env.local_obs()
    check("layout sizes", o.shape == (N_AGENTS, LOCAL_DIM)
          and env.global_state().shape == (GLOBAL_DIM,))


def test_sighted_technique():
    disc = tech("Discovery")
    env = EngagementEnv(CFG).reset(spec("S2", [disc, disc], move={0: 0.0}))
    env.step(joint())
    o = env.local_obs()
    check("only the sighted agent sees the technique",
          o[AG["DMZ"], O_SIGHTED] == 1 and o[AG["DMZ"], O_TECH + disc] == 1
          and o[[AG["LAN"], AG["User"], AG["Admin"]], O_TECH:O_TECH + 84].sum() == 0)


def test_determinism_crn_peek():
    a = EpisodeSource(7).take(5)
    b = EpisodeSource(7).take(5)
    check("episode stream is deterministic in its seed",
          all(x.script == y.script and np.array_equal(x.tau, y.tau)
              and np.array_equal(x.u, y.u) for x, y in zip(a, b)))
    rng = np.random.default_rng(1)
    env = EngagementEnv(CFG)
    mismatches, steps = 0, 0
    for sp in EpisodeSource(99).take(200):
        env.reset(sp)
        while not env.done:
            want = env.peek_next_zone()
            _, _, info = env.step(list(rng.integers(0, 6, N_AGENTS)))
            mismatches += int(info["zone"] != want); steps += 1
    check("peek_next_zone() predicts the move exactly (clairvoyant oracles)",
          mismatches == 0, "%d steps" % steps)

    class Const:
        def __init__(self, a):
            self.a = a

        def __call__(self, env):
            return self.a
    s1 = [run_episode(EngagementEnv(CFG), sp, Const(joint(DMZ=1, LAN=2)))
          for sp in EpisodeSource(5).take(50)]
    s2 = [run_episode(EngagementEnv(CFG), sp, Const(joint(DMZ=1, LAN=2)))
          for sp in EpisodeSource(5).take(50)]
    check("common random numbers: identical actions -> identical outcomes",
          s1 == s2)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 88)
    print("  TEMARL v7 — environment unit tests  (frozen fingerprint %s)"
          % _frozen.fingerprint())
    print("=" * 88)
    for fn in (test_scripts, test_capacity, test_breadcrumb_chain, test_dead_end,
               test_exposure_hazard, test_progress_rules, test_horizon_and_reward,
               test_locality, test_sighted_technique, test_determinism_crn_peek):
        print("\n %s" % fn.__name__)
        fn()
    n, k = len(RESULTS), sum(RESULTS)
    print("\n  %d / %d passed" % (k, n))
    print("=" * 88)
    sys.exit(0 if k == n else 1)


if __name__ == "__main__":
    main()
