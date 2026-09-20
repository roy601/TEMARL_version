# -*- coding: utf-8 -*-
"""
TEMARL v7 — history-encoder pretraining (then FROZEN) and the h bank
====================================================================
Every encoder arm is trained on the SAME task, data, budget and optimiser:
predict the attacker's next technique from the SIEM log of the techniques it
has executed so far (a right-padded window of the last 16). The recipe is the
COMISET one (AdamW, lr 5e-4, weight decay 1e-4, batch 256, 3000 steps), used
unchanged for every arm and never tuned per arm.

The encoder is then FROZEN before any policy learning: the representation must
not chase the value function (two-timescale argument, as in v2-v6). Because the
attacker's technique sequence is exogenous in env_marl, the embedding h for
every decision point of an episode can be computed in one batched pass when
the episode is drawn (`HistoryBank`).

For leave-one-script-out generalisation the encoder sees ONLY the training
scripts; the held-out script is never in its data.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _frozen  # noqa: F401
from encoders_marl import H_DIM, build_encoder
from env_marl import EpisodeSource, EpisodeSpec
from vocab_v2 import MAX_SEQ_LEN, NUM_TECHNIQUES, PAD_ID

PRETRAIN = {"episodes": 3000, "steps": 3000, "batch": 256, "lr": 5e-4,
            "weight_decay": 1e-4, "val_episodes": 400}


def windows(specs: Sequence[EpisodeSpec]):
    """(X, L, Y): for every decision point t >= 1 of every episode, the window
    of techniques 1..t (last MAX_SEQ_LEN, right-padded) and technique t+1."""
    X, L, Y = [], [], []
    for sp in specs:
        tau = sp.tau
        for t in range(1, sp.horizon):
            w = tau[max(0, t - MAX_SEQ_LEN):t]
            row = np.full(MAX_SEQ_LEN, PAD_ID, np.int64)
            row[:len(w)] = w
            X.append(row); L.append(len(w)); Y.append(int(tau[t]))
    return (np.asarray(X, np.int64), np.asarray(L, np.int64),
            np.asarray(Y, np.int64))


def baselines(Ytr, Xtr, Ltr, Yva, Xva, Lva):
    """Majority class and first-order bigram (argmax next | current), fitted on
    the training windows and scored on validation."""
    maj = np.bincount(Ytr, minlength=NUM_TECHNIQUES).argmax()
    cur_tr = Xtr[np.arange(len(Xtr)), Ltr - 1]
    C = np.zeros((NUM_TECHNIQUES + 2, NUM_TECHNIQUES))
    np.add.at(C, (cur_tr, Ytr), 1.0)
    big = C.argmax(1)
    big[C.sum(1) == 0] = maj
    cur_va = Xva[np.arange(len(Xva)), Lva - 1]
    return {"majority": float((Yva == maj).mean()),
            "bigram": float((big[cur_va] == Yva).mean())}


def pretrain(arm: str, seed: int, pool: Sequence[str], device: str = "cpu",
             verbose: bool = False, smoke: bool = False) -> Dict:
    """Train and freeze one encoder. Returns {"encoder", "metrics"}.
    `smoke` shrinks the budget for code-path tests only; never for results."""
    if arm == "NoHistory":
        return {"encoder": None, "metrics": {"arm": arm}}
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    n_ep, n_steps = ((200, 100) if smoke
                     else (PRETRAIN["episodes"], PRETRAIN["steps"]))
    tr = EpisodeSource(60_000 + seed, pool).take(n_ep)
    va = EpisodeSource(70_000 + seed, pool).take(PRETRAIN["val_episodes"])
    Xtr, Ltr, Ytr = windows(tr)
    Xva, Lva, Yva = windows(va)
    enc = build_encoder(arm).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=PRETRAIN["lr"],
                            weight_decay=PRETRAIN["weight_decay"])
    Xt = torch.as_tensor(Xtr, device=device)
    Lt = torch.as_tensor(Ltr, device=device)
    Yt = torch.as_tensor(Ytr, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    enc.train()
    for _ in range(n_steps):
        b = torch.randint(0, len(Ytr), (PRETRAIN["batch"],), generator=g).to(device)
        _, lg = enc.pretrain_forward(Xt[b], Lt[b])
        # PAD/UNK are input sentinels, never valid next-technique targets.
        loss = F.cross_entropy(lg[:, :NUM_TECHNIQUES], Yt[b])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
    enc.freeze()
    with torch.no_grad():
        accs, acc3 = [], []
        for i in range(0, len(Yva), 4096):
            _, lg = enc.pretrain_forward(torch.as_tensor(Xva[i:i + 4096], device=device),
                                         torch.as_tensor(Lva[i:i + 4096], device=device))
            lg = lg[:, :NUM_TECHNIQUES]
            y = torch.as_tensor(Yva[i:i + 4096], device=device)
            accs.append((lg.argmax(-1) == y).float().cpu().numpy())
            acc3.append((lg.topk(3, -1).indices == y[:, None]).any(-1).float().cpu().numpy())
    m = {"arm": arm, "top1": float(np.concatenate(accs).mean()),
         "top3": float(np.concatenate(acc3).mean()),
         **baselines(Ytr, Xtr, Ltr, Yva, Xva, Lva),
         "n_train_windows": int(len(Ytr)), "seconds": time.time() - t0}
    if verbose:
        print("    pretrain %-17s top1 %.3f (majority %.3f, bigram %.3f)  %.0fs"
              % (arm, m["top1"], m["majority"], m["bigram"], m["seconds"]))
    return {"encoder": enc, "metrics": m}


class HistoryBank:
    """Computes the h table of an episode: h[t] embeds techniques 1..t, and
    h[0] = 0 (nothing observed yet). NoHistory returns all-zero tables."""

    def __init__(self, encoder, device: str = "cpu", chunk: int = 8192):
        self.enc, self.device, self.chunk = encoder, device, chunk

    def tables(self, specs: Sequence[EpisodeSpec]) -> List[np.ndarray]:
        out = [np.zeros((sp.horizon, H_DIM), np.float32) for sp in specs]
        if self.enc is None:
            return out
        X, L, idx = [], [], []
        for k, sp in enumerate(specs):
            for t in range(1, sp.horizon):
                w = sp.tau[max(0, t - MAX_SEQ_LEN):t]
                row = np.full(MAX_SEQ_LEN, PAD_ID, np.int64)
                row[:len(w)] = w
                X.append(row); L.append(len(w)); idx.append((k, t))
        if not X:
            return out
        X = np.asarray(X); L = np.asarray(L)
        H = np.empty((len(X), H_DIM), np.float32)
        with torch.no_grad():
            for i in range(0, len(X), self.chunk):
                h = self.enc(torch.as_tensor(X[i:i + self.chunk], device=self.device),
                             torch.as_tensor(L[i:i + self.chunk], device=self.device))
                H[i:i + self.chunk] = h.float().cpu().numpy()
        for (k, t), hv in zip(idx, H):
            out[k][t] = hv
        return out
