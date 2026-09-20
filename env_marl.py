# -*- coding: utf-8 -*-
"""
TEMARL v7 — Coordinated Engagement Deception environment
========================================================
A Dec-POMDP in which four defender agents, one per defended CAM-LDS zone,
deploy D3FEND decoys against an attacker that replays the statistics of a real
CAM-LDS attack script. It differs from `env_v2`/`env_entity` in the two ways
the v7 audit required:

1. ENGAGEMENT, NOT CONTAINMENT. A deceived attacker is not removed: it keeps
   interacting with the decoy, and every such step is an interaction step.
   The episode return is therefore the proposal's DWELL TIME, and deceiving the
   attacker lengthens it instead of ending it. (In env_v2 capture ended the
   episode, which made dwell time fall as the defence improved.)

2. GENUINE MULTI-AGENT COUPLING. In env_v2/env_entity only the responsible
   agent's action entered the reward, so the team problem decomposed into
   independent single-agent problems. Here three mechanisms make one agent's
   best action depend on the others':

   (a) CAPACITY -- substitutes. The deception infrastructure runs at most
       K = 2 decoys at full fidelity. With n_active > K every decoy's fidelity
       drops to phi = K / n_active (resource contention makes decoys slow and
       fingerprintable: Holz & Raynal 2005; Fu et al. 2006).
   (b) BREADCRUMBS -- complements. D3-DUC (decoy credential) and D3-DF (decoy
       file) are breadcrumbs (honeytokens, Spitzner 2003). When the attacker
       engages one, the orchestrator wires it to an active decoy in ANOTHER
       zone and the attacker follows it there next step. With no partner decoy
       the breadcrumb is a dead end that risks exposing the deception. A lure
       is worthless alone and valuable only with a partner.
   (c) SHARED EXPOSURE -- an externality. Once the attacker recognises the
       deception it ignores every decoy for the rest of the episode, so one
       agent's unrealistic decoy costs all agents their future engagement.

PER-STEP SEMANTICS (anticipatory, as in v2/v5: actions are chosen BEFORE the
attacker moves, so an agent must predict where it will go and what it will do)
  1. each agent i chooses d_i in {none, D3-DNR, D3-DE, D3-DUC, D3-DF, D3-DP}
  2. n_active = #{d_i != none};  phi = min(1, K / n_active)
  3. the attacker moves: it follows a pending breadcrumb if one was wired last
     step; otherwise from outside its target zone it heads there with p_goal
     (else wanders to a non-target zone), and inside it stays with p_stay
     (else wanders)
  4. it executes its next technique tau in its new zone z. If z's decoy d is
     active and the deception is not exposed, it engages with probability
     rho = phi * PAYOFF[tau, d]. An engaged step adds one dwell step and adds
     tau to the depth set, touches no real asset, and exposes the deception
     with probability beta * (1 - rho). An engaged breadcrumb is wired to a
     random active decoy in another zone; with none, exposure risk beta.
  5. otherwise tau runs on a real host of z. If z is the target zone and tau
     belongs to the goal tactic, the objective is reached and the episode ends.
  6. the episode also ends when the script's action budget (horizon) is spent.

COMMON RANDOM NUMBERS. Everything random -- the script, its horizon, the
technique sequence, and one uniform per (purpose, step) -- is drawn up front
into an `EpisodeSpec`. Two policies that act identically in the same state
therefore see identical outcomes, which is what makes the paired comparisons
between arms sharp. The technique sequence is exogenous (defender actions
change where techniques land, never which techniques are chosen), so the
shared SIEM history embedding h can be computed per episode in advance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

import _frozen  # noqa: F401
from d3fend_payoff import ACTIONS, N_ACTIONS, PAYOFF
from env_v2 import DEFENDED_ZONES, ZONES
from scripts_marl import max_horizon, script_ids, scripts
from vocab_v2 import NUM_TECHNIQUES

NT = NUM_TECHNIQUES
N_ZONES = len(ZONES)
N_AGENTS = len(DEFENDED_ZONES)
AGENT_ZONE = list(DEFENDED_ZONES)                 # agent i owns AGENT_ZONE[i]
ZONE_AGENT = {z: i for i, z in enumerate(AGENT_ZONE)}
NULL = 0
LURES = (ACTIONS.index("decoy_credential"), ACTIONS.index("decoy_file"))
assert LURES == (3, 4), LURES
MAX_STEPS = max_horizon()                         # 61: the longest real run
SCRIPT_IDS = script_ids()
N_SCRIPTS = len(SCRIPT_IDS)

# uniform-stream rows of an EpisodeSpec
U_MOVE, U_LATERAL, U_ENGAGE, U_BURN, U_DEAD, U_WIRE = range(6)
N_U = 6


@dataclass(frozen=True)
class EnvConfig:
    """Free parameters. K and G are fixed by design; beta / p_goal / p_stay are
    chosen by the architecture-blind calibration (calibrate_marl.py) and then
    frozen to env_config.json."""
    beta: float = 0.3
    p_goal: float = 0.6
    p_stay: float = 0.7
    k_capacity: int = 2
    goal_steps: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class EpisodeSpec:
    script: str
    horizon: int
    tau: np.ndarray            # (horizon,) executed techniques
    u: np.ndarray              # (N_U, horizon) uniforms, one per purpose/step


def sample_episode(rng: np.random.Generator,
                   script_pool: Sequence[str] = SCRIPT_IDS) -> EpisodeSpec:
    """Draw everything random about one episode, up front."""
    sc_all = scripts()
    sid = script_pool[int(rng.integers(len(script_pool)))]
    sc = sc_all[sid]
    horizon = int(sc["horizons"][int(rng.integers(len(sc["horizons"])))])
    u_tau = rng.random(horizon)
    tau = np.empty(horizon, np.int64)
    tau[0] = min(int(np.searchsorted(sc["cuminit"], u_tau[0], side="right")), NT - 1)
    for t in range(1, horizon):
        row = sc["cumT"][tau[t - 1]]
        tau[t] = min(int(np.searchsorted(row, u_tau[t], side="right")), NT - 1)
    u = rng.random((N_U, horizon))
    return EpisodeSpec(sid, horizon, tau, u)


class EpisodeSource:
    """Deterministic stream of episode specs from one seed."""

    def __init__(self, seed: int, script_pool: Sequence[str] = SCRIPT_IDS):
        self.rng = np.random.default_rng(seed)
        self.pool = list(script_pool)

    def next(self) -> EpisodeSpec:
        return sample_episode(self.rng, self.pool)

    def take(self, n: int) -> List[EpisodeSpec]:
        return [self.next() for _ in range(n)]


# ── observation layout (per agent, LOCAL) ────────────────────────────────────
O_AGENT = 0
O_SIGHTED = O_AGENT + N_AGENTS
O_TECH = O_SIGHTED + 1
O_MYDECOY = O_TECH + NT
O_MYENGAGED = O_MYDECOY + N_ACTIONS
O_WIRED = O_MYENGAGED + 1
O_MYCOMP = O_WIRED + 1
O_LOAD = O_MYCOMP + 1
O_STEP = O_LOAD + 1
LOCAL_DIM = O_STEP + 1

# ── global state layout (critic only; centralised training) ─────────────────
S_ZONE = 0
S_TECH = S_ZONE + N_ZONES
S_SCRIPT = S_TECH + NT
S_DECOYS = S_SCRIPT + N_SCRIPTS
S_BURNED = S_DECOYS + N_AGENTS * N_ACTIONS
S_PROGRESS = S_BURNED + 1
S_COMP = S_PROGRESS + 1
S_LOAD = S_COMP + N_ZONES
S_PENDING = S_LOAD + 1
S_STEP = S_PENDING + N_ZONES
S_REMAIN = S_STEP + 1
GLOBAL_DIM = S_REMAIN + 1


class EngagementEnv:
    """One episode at a time; `VecEngagementEnv` batches them."""

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self._sc = scripts()
        self.spec: Optional[EpisodeSpec] = None
        self.done = True

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self, spec: EpisodeSpec):
        sc = self._sc[spec.script]
        self.spec = spec
        self.script = spec.script
        self.script_idx = SCRIPT_IDS.index(spec.script)
        self.target = sc["target"]
        self.goal_mask = sc["goal_mask"]
        self.zone = sc["entry"]
        self.t = 0
        self.burned = False
        self.pending = -1                  # zone a wired breadcrumb leads to
        self.decoys = np.zeros(N_AGENTS, np.int64)
        self.my_engaged = np.zeros(N_AGENTS, bool)
        self.compromised = np.zeros(N_ZONES, np.float32)
        self.load = 0
        self.progress = 0
        self.cur_tau = -1                  # technique executed at the last step
        self.done = False
        self.objective = False
        self._chain_open = False
        # episode statistics
        self.dwell = 0
        self.depth_set = set()
        self.st = {"capacity_violations": 0, "lure_engagements": 0,
                   "lure_chains": 0, "dead_ends": 0, "decoys_deployed": 0,
                   "engaged_in_target": 0, "exposed_at": -1}
        return self

    # ── dynamics ────────────────────────────────────────────────────────────
    def _move(self, u_move: float, u_lat: float) -> int:
        cur, tgt = self.zone, self.target
        if cur == tgt:
            if u_move < self.cfg.p_stay:
                return cur
            cands = [z for z in AGENT_ZONE if z != cur]
        else:
            if u_move < self.cfg.p_goal:
                return tgt
            cands = [z for z in AGENT_ZONE if z != tgt]
        return cands[min(int(u_lat * len(cands)), len(cands) - 1)]

    def peek_next_zone(self) -> int:
        """Where the attacker WILL be after the next step (clairvoyant oracles
        only -- it reads the pre-drawn uniforms). Pure: no state change."""
        if self.pending >= 0:
            return self.pending
        u = self.spec.u[:, self.t]
        return self._move(u[U_MOVE], u[U_LATERAL])

    def step(self, actions: Sequence[int]):
        if self.done:
            raise RuntimeError("step() on a finished episode; reset() first")
        acts = np.asarray(actions, dtype=np.int64)
        if acts.shape != (N_AGENTS,) or acts.min() < 0 or acts.max() >= N_ACTIONS:
            raise ValueError(f"bad joint action {actions}")
        t = self.t
        tau = int(self.spec.tau[t])
        u = self.spec.u[:, t]
        cfg = self.cfg

        n_active = int((acts != NULL).sum())
        phi = 1.0 if n_active <= cfg.k_capacity else cfg.k_capacity / n_active

        # 1. move (a wired breadcrumb overrides the script's own movement)
        followed = self.pending >= 0
        nz = self.pending if followed else self._move(u[U_MOVE], u[U_LATERAL])
        self.pending = -1
        self.zone = nz
        ag = ZONE_AGENT[nz]                # after a move the zone is defended
        d = int(acts[ag])

        # 2. engagement
        engaged = exposed = False
        wired = -1
        rho = 0.0
        if d != NULL and not self.burned:
            rho = phi * float(PAYOFF[tau, d])
            if u[U_ENGAGE] < rho:
                engaged = True
                if u[U_BURN] < cfg.beta * (1.0 - rho):
                    exposed = True
                if d in LURES:
                    self.st["lure_engagements"] += 1
                    partners = [z for z in AGENT_ZONE
                                if z != nz and acts[ZONE_AGENT[z]] != NULL]
                    if partners:
                        wired = partners[min(int(u[U_WIRE] * len(partners)),
                                             len(partners) - 1)]
                    else:
                        self.st["dead_ends"] += 1
                        if u[U_DEAD] < cfg.beta:
                            exposed = True
        if exposed:
            self.burned = True
            wired = -1
            self.st["exposed_at"] = t

        if engaged:
            self.dwell += 1
            self.depth_set.add(tau)
            if nz == self.target:
                self.st["engaged_in_target"] += 1
            if followed and self._chain_open:
                self.st["lure_chains"] += 1
        else:
            self.compromised[nz] = 1.0
            if nz == self.target and self.goal_mask[tau]:
                self.progress += 1
        self._chain_open = wired >= 0
        if wired >= 0:
            self.pending = wired

        # 3. bookkeeping visible to the agents next step
        self.decoys = acts.copy()
        self.my_engaged[:] = False
        if engaged:
            self.my_engaged[ag] = True
        self.load = n_active
        self.cur_tau = tau
        self.st["decoys_deployed"] += n_active
        if n_active > cfg.k_capacity:
            self.st["capacity_violations"] += 1
        self.t += 1

        # 4. termination
        if self.progress >= cfg.goal_steps:
            self.objective = True
            self.done = True
        elif self.t >= self.spec.horizon:
            self.done = True

        info = {"engaged": engaged, "exposed": exposed, "zone": nz,
                "tau": tau, "rho": rho, "phi": phi, "wired": wired,
                "followed": followed}
        return (1.0 if engaged else 0.0), self.done, info

    # ── observations ────────────────────────────────────────────────────────
    def local_obs(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """(N_AGENTS, LOCAL_DIM). Each row holds ONLY what that agent can see:
        its own zone, its own decoy, and two infrastructure signals (last
        step's load, and whether a breadcrumb now points at its decoy)."""
        o = np.zeros((N_AGENTS, LOCAL_DIM), np.float32) if out is None else out
        if out is not None:
            o[:] = 0.0
        for i, z in enumerate(AGENT_ZONE):
            o[i, O_AGENT + i] = 1.0
            if self.zone == z:
                o[i, O_SIGHTED] = 1.0
                if self.cur_tau >= 0:
                    o[i, O_TECH + self.cur_tau] = 1.0
            o[i, O_MYDECOY + int(self.decoys[i])] = 1.0
            o[i, O_MYENGAGED] = float(self.my_engaged[i])
            o[i, O_WIRED] = 1.0 if self.pending == z else 0.0
            o[i, O_MYCOMP] = self.compromised[z]
            o[i, O_LOAD] = self.load / self.cfg.k_capacity
            o[i, O_STEP] = self.t / MAX_STEPS
        return o

    def global_state(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        g = np.zeros(GLOBAL_DIM, np.float32) if out is None else out
        if out is not None:
            g[:] = 0.0
        g[S_ZONE + self.zone] = 1.0
        if self.cur_tau >= 0:
            g[S_TECH + self.cur_tau] = 1.0
        g[S_SCRIPT + self.script_idx] = 1.0
        for i in range(N_AGENTS):
            g[S_DECOYS + i * N_ACTIONS + int(self.decoys[i])] = 1.0
        g[S_BURNED] = float(self.burned)
        g[S_PROGRESS] = self.progress / self.cfg.goal_steps
        g[S_COMP:S_COMP + N_ZONES] = self.compromised
        g[S_LOAD] = self.load / self.cfg.k_capacity
        if self.pending >= 0:
            g[S_PENDING + self.pending] = 1.0
        g[S_STEP] = self.t / MAX_STEPS
        g[S_REMAIN] = (self.spec.horizon - self.t) / MAX_STEPS
        return g

    def history(self) -> np.ndarray:
        """Techniques executed so far (the SIEM log h is computed from)."""
        return self.spec.tau[:self.t]

    def episode_stats(self) -> Dict[str, float]:
        return {"dwell": float(self.dwell),
                "depth": float(len(self.depth_set)),
                "protected": float(not self.objective),
                "exposed": float(self.burned),
                "length": float(self.t),
                "horizon": float(self.spec.horizon),
                "script": self.script,
                **{k: float(v) for k, v in self.st.items()}}


def run_episode(env: EngagementEnv, spec: EpisodeSpec, policy) -> Dict[str, float]:
    """Roll one episode with `policy(env) -> joint action` (baselines/oracles)."""
    env.reset(spec)
    if hasattr(policy, "begin"):
        policy.begin(env)
    while not env.done:
        env.step(policy(env))
    return env.episode_stats()
