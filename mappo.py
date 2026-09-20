# -*- coding: utf-8 -*-
"""
TEMARL v7 — MAPPO (and IPPO) for the coordinated engagement environment
=======================================================================
Multi-Agent PPO with centralised training and decentralised execution
(Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games", NeurIPS 2022 Datasets & Benchmarks):

  ACTOR   decentralised and parameter-shared. Agent i sees ONLY its local
          observation (its zone, its decoy, two infrastructure signals; the
          agent id is part of it) and the shared SIEM intent embedding h. The
          fusion is the project's GMU trunk (`policy.GatedFusionTrunk`,
          Arevalo et al. 2017), unchanged, followed by a categorical head over
          the six D3FEND decoy types.
  CRITIC  centralised: V(global state, h, agent id). Used only in training --
          execution never touches it (CTDE).
  IPPO    the ablation: the same actor with a LOCAL critic V(local obs, h)
          (de Witt et al. 2020). MAPPO vs IPPO tests whether centralised
          training matters in this coupled environment.

Hyperparameters follow the MAPPO paper's recommendations (value normalisation,
clip 0.2, one mini-batch, <= 15 epochs, entropy 0.01, Huber value loss,
gradient-norm 10, orthogonal/xavier init) and are IDENTICAL for every arm.
They are not tuned per arm. The only quantity the learning gate may choose is
the training budget, and it chooses it on the GRU arm alone.

Checkpoint selection uses a VALIDATION episode stream that is disjoint from the
TEST stream every reported number comes from.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _frozen
from d3fend_payoff import N_ACTIONS
from encoders_marl import H_DIM
from env_marl import (GLOBAL_DIM, LOCAL_DIM, N_AGENTS, O_SIGHTED, SCRIPT_IDS,
                      EngagementEnv, EnvConfig, EpisodeSource, EpisodeSpec)
from policy import BRANCH_DIM, GatedFusionTrunk
from pretrain_marl import HistoryBank

HP = {
    "lr": 5e-4, "ppo_epochs": 10, "n_minibatch": 1, "clip": 0.2,
    "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.01,
    "max_grad_norm": 10.0, "huber_delta": 10.0,
    "n_envs": 64, "rollout_len": 64,
    "critic_hidden": 128, "head_hidden": 64, "gate_bias": 2.0,
    "n_evals": 10, "val_episodes": 300,
}
SEED_TRAIN, SEED_VAL, SEED_TEST = 100_000, 200_000, 300_000
EYE = np.eye(N_AGENTS, dtype=np.float32)


# ── networks ─────────────────────────────────────────────────────────────────

def _ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = GatedFusionTrunk(state_dim=LOCAL_DIM, h_dim=H_DIM,
                                      gate_bias=HP["gate_bias"],
                                      sighted_idx=O_SIGHTED)
        self.head = nn.Sequential(
            _ortho(nn.Linear(BRANCH_DIM, HP["head_hidden"]), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Linear(HP["head_hidden"], N_ACTIONS), 0.01))

    def forward(self, local, h):
        f, gate, _ = self.trunk(torch.cat([local, h], -1))
        return self.head(f), gate


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        H = HP["critic_hidden"]
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            _ortho(nn.Linear(in_dim, H), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Linear(H, H), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Linear(H, 1), 1.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ValueNorm:
    """Running normalisation of value targets (MAPPO's ValueNorm)."""

    def __init__(self, beta=0.99999, eps=1e-5):
        self.beta, self.eps = beta, eps
        self.m = self.m2 = self.debias = 0.0

    def update(self, x: torch.Tensor):
        # statistics in float64 on the CPU: identical to the CPU runs, and safe
        # on GPUs without float64 support (e.g. Intel Arc)
        x = x.detach().cpu().double()
        self.m = self.beta * self.m + (1 - self.beta) * float(x.mean())
        self.m2 = self.beta * self.m2 + (1 - self.beta) * float((x ** 2).mean())
        self.debias = self.beta * self.debias + (1 - self.beta)

    def stats(self):
        mean = self.m / max(self.debias, self.eps)
        var = max(self.m2 / max(self.debias, self.eps) - mean ** 2, 1e-2)
        return mean, var ** 0.5

    def normalize(self, x):
        mu, sd = self.stats()
        return (x - mu) / sd

    def denormalize(self, x):
        mu, sd = self.stats()
        return x * sd + mu


def critic_input(centralised: bool, local, h, glob):
    """local (N, A, LOCAL_DIM), h (N, H), glob (N, GLOBAL_DIM) -> (N*A, in)."""
    n = local.shape[0]
    hr = np.repeat(h[:, None, :], N_AGENTS, 1)
    if centralised:
        g = np.repeat(glob[:, None, :], N_AGENTS, 1)
        ids = np.broadcast_to(EYE, (n, N_AGENTS, N_AGENTS))
        x = np.concatenate([g, hr, ids], -1)
    else:
        x = np.concatenate([local, hr], -1)
    return x.reshape(n * N_AGENTS, -1)


def critic_dim(centralised: bool) -> int:
    return (GLOBAL_DIM + H_DIM + N_AGENTS) if centralised else (LOCAL_DIM + H_DIM)


# ── batched environment with a prefetching episode source ───────────────────

class Episodes:
    """Streams (spec, h_table) pairs, computing h in large batches."""

    def __init__(self, source: EpisodeSource, bank: HistoryBank, chunk=256):
        self.source, self.bank, self.chunk = source, bank, chunk
        self.q = deque()

    def next(self):
        if not self.q:
            specs = self.source.take(self.chunk)
            for sp, tb in zip(specs, self.bank.tables(specs)):
                self.q.append((sp, tb))
        return self.q.popleft()


class VecEnv:
    def __init__(self, n: int, cfg: EnvConfig, episodes: Episodes):
        self.envs = [EngagementEnv(cfg) for _ in range(n)]
        self.episodes = episodes
        self.h = [None] * n
        self.local = np.zeros((n, N_AGENTS, LOCAL_DIM), np.float32)
        self.glob = np.zeros((n, GLOBAL_DIM), np.float32)
        self.hcur = np.zeros((n, H_DIM), np.float32)
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        sp, tb = self.episodes.next()
        self.envs[i].reset(sp)
        self.h[i] = tb
        self._obs(i)

    def _obs(self, i):
        e = self.envs[i]
        e.local_obs(out=self.local[i])
        e.global_state(out=self.glob[i])
        self.hcur[i] = self.h[i][e.t] if e.t < len(self.h[i]) else 0.0

    def step(self, actions: np.ndarray):
        n = len(self.envs)
        rew = np.zeros(n, np.float32)
        done = np.zeros(n, np.float32)
        finished = []
        for i, e in enumerate(self.envs):
            r, d, _ = e.step(actions[i])
            rew[i] = r
            if d:
                done[i] = 1.0
                finished.append(e.episode_stats())
                self._reset(i)
            else:
                self._obs(i)
        return rew, done, finished


# ── evaluation (greedy, decentralised) ──────────────────────────────────────

@torch.no_grad()
def evaluate_team(actors, cfg: EnvConfig, specs: Sequence[EpisodeSpec], bank: HistoryBank,
                  device="cpu", h_zero=False, greedy=True, batch=128,
                  seed=0) -> List[Dict]:
    """Roll every spec to the end. `actors` is one Actor (parameter sharing) or
    a list of N_AGENTS actors (cross-play: agent i acts with actors[i])."""
    if not isinstance(actors, (list, tuple)):
        actors = [actors] * N_AGENTS
    for a in actors:
        a.eval()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    out = [None] * len(specs)
    tables = bank.tables(specs)
    for s0 in range(0, len(specs), batch):
        idx = list(range(s0, min(len(specs), s0 + batch)))
        envs = [EngagementEnv(cfg).reset(specs[k]) for k in idx]
        local = np.zeros((len(idx), N_AGENTS, LOCAL_DIM), np.float32)
        while True:
            live = [j for j, e in enumerate(envs) if not e.done]
            if not live:
                break
            for j in live:
                envs[j].local_obs(out=local[j])
            L = torch.as_tensor(local[live], device=device)
            H = np.stack([np.zeros(H_DIM, np.float32) if h_zero
                          else tables[idx[j]][envs[j].t] for j in live])
            Ht = torch.as_tensor(H, device=device)
            acts = np.zeros((len(live), N_AGENTS), np.int64)
            for ag in range(N_AGENTS):
                logits, _ = actors[ag](L[:, ag], Ht)
                if greedy:
                    acts[:, ag] = logits.argmax(-1).cpu().numpy()
                else:
                    acts[:, ag] = torch.multinomial(
                        logits.cpu().softmax(-1), 1, generator=rng).squeeze(-1).numpy()
            for k, j in enumerate(live):
                envs[j].step(acts[k])
        for j, k in enumerate(idx):
            out[k] = envs[j].episode_stats()
    return out


def mean_of(rows, key="dwell"):
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


# ── training ────────────────────────────────────────────────────────────────

def train(encoder, cfg: EnvConfig, pool: Sequence[str], seed: int,
          env_steps: int, centralised: bool = True, device: str = "cpu",
          test_pool: Optional[Sequence[str]] = None, n_test: int = 500,
          test_seed: Optional[int] = None,
          verbose: bool = False, log_every: int = 25) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    bank = HistoryBank(encoder, device)
    vec = VecEnv(HP["n_envs"], cfg,
                 Episodes(EpisodeSource(SEED_TRAIN + seed, pool), bank))
    actor = Actor().to(device)
    critic = Critic(critic_dim(centralised)).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=HP["lr"], eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=HP["lr"], eps=1e-5)
    vnorm = ValueNorm()
    val_specs = EpisodeSource(SEED_VAL + seed, pool).take(HP["val_episodes"])

    T, E, A = HP["rollout_len"], HP["n_envs"], N_AGENTS
    n_updates = max(1, env_steps // (T * E))
    eval_at = set(int(round(n_updates * k / HP["n_evals"]))
                  for k in range(1, HP["n_evals"] + 1))
    curve, recent = [], deque(maxlen=400)
    best = {"val": -1.0, "actor": None, "update": 0}

    for upd in range(1, n_updates + 1):
        buf_l = np.zeros((T, E, A, LOCAL_DIM), np.float32)
        buf_h = np.zeros((T, E, H_DIM), np.float32)
        buf_c = np.zeros((T, E * A, critic_dim(centralised)), np.float32)
        buf_a = np.zeros((T, E, A), np.int64)
        buf_lp = np.zeros((T, E, A), np.float32)
        buf_v = np.zeros((T, E, A), np.float32)
        buf_r = np.zeros((T, E), np.float32)
        buf_d = np.zeros((T, E), np.float32)
        actor.eval(); critic.eval()
        with torch.no_grad():
            for t in range(T):
                buf_l[t], buf_h[t] = vec.local, vec.hcur
                buf_c[t] = critic_input(centralised, vec.local, vec.hcur, vec.glob)
                Lt = torch.as_tensor(buf_l[t].reshape(E * A, LOCAL_DIM), device=device)
                Ht = torch.as_tensor(np.repeat(buf_h[t], A, 0), device=device)
                logits, _ = actor(Lt, Ht)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                buf_lp[t] = dist.log_prob(act).cpu().numpy().reshape(E, A)
                buf_v[t] = critic(torch.as_tensor(buf_c[t], device=device)).cpu().numpy().reshape(E, A)
                buf_a[t] = act.cpu().numpy().reshape(E, A)
                r, d, fin = vec.step(buf_a[t])
                buf_r[t], buf_d[t] = r, d
                for st in fin:
                    recent.append(st["dwell"])
            v_last = critic(torch.as_tensor(
                critic_input(centralised, vec.local, vec.hcur, vec.glob),
                device=device)).cpu().numpy().reshape(E, A)

        # GAE on the de-normalised value scale
        vals = np.asarray(vnorm.denormalize(buf_v), np.float32) if vnorm.debias > 0 else buf_v
        vlast = np.asarray(vnorm.denormalize(v_last), np.float32) if vnorm.debias > 0 else v_last
        adv = np.zeros((T, E, A), np.float32)
        gae = np.zeros((E, A), np.float32)
        for t in reversed(range(T)):
            nxt = vlast if t == T - 1 else vals[t + 1]
            nonterm = (1.0 - buf_d[t])[:, None]
            delta = buf_r[t][:, None] + HP["gamma"] * nxt * nonterm - vals[t]
            gae = delta + HP["gamma"] * HP["gae_lambda"] * nonterm * gae
            adv[t] = gae
        ret = adv + vals

        # PPO update
        actor.train(); critic.train()
        N = T * E * A
        Lb = torch.as_tensor(buf_l.reshape(N, LOCAL_DIM), device=device)
        Hb = torch.as_tensor(np.repeat(buf_h.reshape(T * E, H_DIM), A, 0), device=device)
        Cb = torch.as_tensor(buf_c.reshape(N, -1), device=device)
        Ab = torch.as_tensor(buf_a.reshape(N), device=device)
        LPb = torch.as_tensor(buf_lp.reshape(N), device=device)
        Vold = torch.as_tensor(buf_v.reshape(N), device=device)
        Rb = torch.as_tensor(ret.reshape(N), device=device)
        vnorm.update(Rb)
        Rn = torch.as_tensor(np.asarray(vnorm.normalize(Rb.cpu().numpy()), np.float32),
                             device=device)
        advb = torch.as_tensor(adv.reshape(N), device=device)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        mb = N // HP["n_minibatch"]
        for _ in range(HP["ppo_epochs"]):
            perm = torch.randperm(N, device=device)
            for k in range(HP["n_minibatch"]):
                ix = perm[k * mb:(k + 1) * mb]
                logits, _ = actor(Lb[ix], Hb[ix])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(Ab[ix])
                ratio = torch.exp(lp - LPb[ix])
                s1 = ratio * advb[ix]
                s2 = torch.clamp(ratio, 1 - HP["clip"], 1 + HP["clip"]) * advb[ix]
                loss_pi = -torch.min(s1, s2).mean() - HP["entropy_coef"] * dist.entropy().mean()
                opt_a.zero_grad(); loss_pi.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), HP["max_grad_norm"])
                opt_a.step()
                v = critic(Cb[ix])
                vc = Vold[ix] + torch.clamp(v - Vold[ix], -HP["clip"], HP["clip"])
                loss_v = torch.max(F.huber_loss(v, Rn[ix], delta=HP["huber_delta"], reduction="none"),
                                   F.huber_loss(vc, Rn[ix], delta=HP["huber_delta"], reduction="none")).mean()
                opt_c.zero_grad(); loss_v.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), HP["max_grad_norm"])
                opt_c.step()

        if upd in eval_at:
            rows = evaluate_team(actor, cfg, val_specs, bank, device)
            vd = mean_of(rows)
            curve.append({"update": upd, "env_steps": upd * T * E,
                          "train_dwell": float(np.mean(recent)) if recent else 0.0,
                          "val_dwell": vd, "val_depth": mean_of(rows, "depth"),
                          "seconds": time.time() - t0})
            if vd > best["val"]:
                best = {"val": vd, "actor": copy.deepcopy(actor.state_dict()),
                        "update": upd}
            if verbose:
                print("      upd %4d/%d  steps %8d  train dwell %6.2f  val dwell %6.2f "
                      "depth %5.2f  (%.0fs)" % (upd, n_updates, upd * T * E,
                                                curve[-1]["train_dwell"], vd,
                                                curve[-1]["val_depth"], time.time() - t0))
        elif verbose and upd % log_every == 0:
            print("      upd %4d/%d  train dwell %6.2f  (%.0fs)"
                  % (upd, n_updates, float(np.mean(recent)) if recent else 0.0,
                     time.time() - t0))

    actor.load_state_dict(best["actor"])
    # the caller passes the SAME test seed it uses for the baselines, so every
    # learned-vs-baseline comparison is on identical episodes (paired)
    test_specs = EpisodeSource(SEED_TEST + seed if test_seed is None else test_seed,
                               test_pool or pool).take(n_test)
    test = evaluate_team(actor, cfg, test_specs, bank, device)
    test_h0 = evaluate_team(actor, cfg, test_specs, bank, device, h_zero=True)
    return {"actor": actor, "critic": critic, "curve": curve,
            "best_update": best["update"], "val_best": best["val"],
            "test_rows": test, "test_rows_h0": test_h0,
            "n_updates": n_updates, "env_steps": n_updates * T * E,
            "seconds": time.time() - t0,
            "actor_params": sum(p.numel() for p in actor.parameters()),
            "critic_params": sum(p.numel() for p in critic.parameters())}


def summary(rows: List[Dict]) -> Dict[str, float]:
    keys = ("dwell", "depth", "protected", "exposed", "lure_chains",
            "dead_ends", "capacity_violations", "length", "decoys_deployed")
    return {k: mean_of(rows, k) for k in keys}


if __name__ == "__main__":
    from encoders_marl import ARMS
    from pretrain_marl import pretrain
    ap = argparse.ArgumentParser(description="one MAPPO/IPPO training run")
    ap.add_argument("--arm", default="GRU", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--ippo", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pretraining budget; code-path test only")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "env_config.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = EnvConfig(**json.load(open(args.config, encoding="utf-8"))["config"]) \
        if os.path.exists(args.config) else EnvConfig()
    pre = pretrain(args.arm, args.seed, SCRIPT_IDS, args.device, verbose=True,
                   smoke=args.smoke)
    out = train(pre["encoder"], cfg, SCRIPT_IDS, args.seed, args.steps,
                centralised=not args.ippo, device=args.device, verbose=True)
    print("  test:", {k: round(v, 3) for k, v in summary(out["test_rows"]).items()})
    print("  test (h=0):", {k: round(v, 3) for k, v in summary(out["test_rows_h0"]).items()})
    print("  %.0f s, actor %d params, critic %d params"
          % (out["seconds"], out["actor_params"], out["critic_params"]))
