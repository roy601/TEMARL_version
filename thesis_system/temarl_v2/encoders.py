# -*- coding: utf-8 -*-
"""
TEMARL v2 — Interchangeable Intent Encoders (Model A / B / C)
=============================================================
Closes F-ARC-05 (no architectural control), F-ARC-06 (GRU PAD contamination),
F-MTH-01 (wrong pooling equation), F-MTH-07 (positional clock leak),
F-ARC-04 (stochastic h from live dropout).

WHY THIS MODULE EXISTS
----------------------
v1 shipped a Transformer with NO encoder-family control, so the thesis's central
architectural claim ("Transformers are superior for kill-chain sequence
modelling") was asserted from unrelated NIDS literature and never tested in-task.
The v1 manuscript itself lists an LSTM comparison as future work (Sec. 6.4). With
COMISET sequences averaging ~5 techniques after run-collapse, a set encoder may
well match self-attention -- and that is a publishable finding either way.

Three encoders, IDENTICAL public interface, capacity-matched within +-10%:

  MODEL A  TransformerIntentEncoder  self-attention, absolute sinusoidal PE
  MODEL B  GRUIntentEncoder          recurrent, packed (no PAD contamination)
  MODEL C  SetIntentEncoder          ORDER-AGNOSTIC bag-of-techniques

Model C is the decisive control. It cannot see order at all. If Model A does not
beat Model C, then order carries no exploitable information in this corpus and
the "sequence modelling" framing must be withdrawn -- an honest, reportable result.

THE PAD / POSITION FIX (F-MTH-07, F-MTH-01)
-------------------------------------------
v1 LEFT-padded (`[PAD]*pad_len + seq`) and pooled at a FIXED final index. With
ABSOLUTE sinusoidal positional encoding, identical sub-sequences therefore
received DIFFERENT positional encodings depending on how much padding preceded
them, so absolute position became a proxy for episode step. This is the
mechanism behind the measured h -> step-index probe at R^2 = 0.97-0.98.

v2 consumes (RIGHT-padded ids, true_length): real tokens always occupy positions
0..n-1, so a given prefix always receives the same encoding, and pooling gathers
index (true_length - 1). test_pad_invariance() below asserts this directly --
padding a sequence must not change h by even 1e-6. v1 would fail that test.

v1's printed Eq 4.3 used `argmax(token != PAD)`, which returns the FIRST maximal
index -- i.e. the FIRST real token under left-padding, the opposite of the
intended "last token pooling". The equation was wrong even where the code was not.
"""

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab_v2 import MAX_SEQ_LEN, PAD_ID, VOCAB_SIZE

warnings.filterwarnings("ignore", message=".*nested tensors.*")

D_MODEL = 64          # fixed across all three: obs layout depends on h_dim=64
DROPOUT = 0.1


# ── Shared base: the interface every encoder must satisfy ────────────────────

class IntentEncoder(nn.Module):
    """Common interface. Any subclass is a drop-in replacement for any other.

    Subclasses implement ONLY `_trunk(emb, lengths, pad_mask) -> (B, D_MODEL)`.
    Everything else -- embedding, pooling contract, prediction head, temperature
    calibration, numpy helpers -- is shared, which is what makes the A/B/C
    comparison a controlled experiment rather than three different pipelines.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL,
                 max_len=MAX_SEQ_LEN, dropout=DROPOUT, pad_id=PAD_ID):
        super().__init__()
        self.d_model, self.max_len, self.pad_id = d_model, max_len, pad_id
        self.vocab_size = vocab_size
        self.temperature = 1.0
        # padding_idx keeps the PAD row at zero and blocks gradient to it
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pretrain_head = nn.Linear(d_model, vocab_size)

    # -- to be provided by subclasses --------------------------------------
    def _trunk(self, emb, lengths, pad_mask):
        raise NotImplementedError

    def trunk_parameters(self):
        """Params excluding embedding + head, for fair capacity matching."""
        shared = set(id(p) for p in self.embedding.parameters()) | \
                 set(id(p) for p in self.pretrain_head.parameters())
        return [p for p in self.parameters() if id(p) not in shared]

    # -- shared forward contract -------------------------------------------
    def forward(self, token_ids, lengths=None):
        """token_ids: (B, L) RIGHT-padded.  lengths: (B,) true lengths.
        Returns h: (B, d_model)."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        B, L = token_ids.shape
        if lengths is None:                       # infer from the pad pattern
            lengths = (token_ids != self.pad_id).sum(1)
        lengths = torch.as_tensor(lengths, device=token_ids.device).long().clamp(min=1)

        # all-PAD guard: an episode's first observation has no history yet.
        # Keep position 0 unmasked so attention/packing cannot receive an empty
        # sequence; h is uninformative there by construction, not by accident.
        pad_mask = (token_ids == self.pad_id)
        pad_mask[:, 0] = False

        emb = self.embedding(token_ids) * math.sqrt(self.d_model)
        return self._trunk(emb, lengths, pad_mask)

    @staticmethod
    def gather_last_real(seq_out, lengths):
        """F-MTH-01: gather the LAST REAL token at index (length-1).
        Correct form of v1's Eq 4.3, which used argmax and returned the FIRST."""
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, seq_out.size(-1))
        return seq_out.gather(1, idx).squeeze(1)

    def encode(self, token_ids, lengths=None):
        return self.forward(token_ids, lengths)

    def pretrain_forward(self, token_ids, lengths=None):
        """Returns (h, logits) — identical tuple order across all encoders."""
        h = self.forward(token_ids, lengths)
        return h, self.pretrain_head(h)

    def set_temperature(self, t):
        self.temperature = float(max(t, 1e-3))

    def calibrated_entropy(self, logits):
        p = torch.softmax(logits / self.temperature, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(-1)
        return ent / math.log(self.vocab_size)

    # -- numpy helpers used inside the env loop ----------------------------
    def get_h_numpy(self, ids_np, device, lengths=None):
        single = (np.asarray(ids_np).ndim == 1)
        arr = np.asarray(ids_np)[None, :] if single else np.asarray(ids_np)
        ln = None if lengths is None else np.atleast_1d(lengths)
        with torch.no_grad():
            t = torch.as_tensor(arr, dtype=torch.long, device=device)
            lt = None if ln is None else torch.as_tensor(ln, dtype=torch.long, device=device)
            h = self.forward(t, lt).cpu().numpy()
        return h[0] if single else h

    def get_h_and_entropy_numpy(self, ids_np, device, lengths=None):
        single = (np.asarray(ids_np).ndim == 1)
        arr = np.asarray(ids_np)[None, :] if single else np.asarray(ids_np)
        ln = None if lengths is None else np.atleast_1d(lengths)
        with torch.no_grad():
            t = torch.as_tensor(arr, dtype=torch.long, device=device)
            lt = None if ln is None else torch.as_tensor(ln, dtype=torch.long, device=device)
            h, lg = self.pretrain_forward(t, lt)
            e = self.calibrated_entropy(lg).cpu().numpy()
            h = h.cpu().numpy()
        return (h[0], float(e[0])) if single else (h, e)

    def freeze(self):
        """F-ARC-03/04: no TD gradients AND eval() so dropout is OFF.
        v1 called encoder.train() during QMIX, so h was stochastic on every
        forward pass; combined with TD gradients into the encoder this made the
        Q-network's input doubly non-stationary and TD loss rose 4.3 -> 7.9
        while the h-frozen ablation converged 9.6 -> 0.84."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


# ── MODEL A — Transformer (the proposed architecture) ────────────────────────

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerIntentEncoder(IntentEncoder):
    """MODEL A. Shallow last-token-pooling encoder per FlowTransformer [8],
    architecture per Vaswani et al. [19]. Absolute sinusoidal PE is SAFE here
    only because inputs are RIGHT-padded (see module docstring)."""

    NAME = "Transformer"

    def __init__(self, n_layers=2, n_heads=4, d_ff=128, **kw):
        super().__init__(**kw)
        self.pos = _SinusoidalPE(self.d_model, self.max_len, DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=DROPOUT, batch_first=True, norm_first=False)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         norm=nn.LayerNorm(self.d_model))

    def _trunk(self, emb, lengths, pad_mask):
        x = self.pos(emb)
        out = self.enc(x, src_key_padding_mask=pad_mask)
        return self.gather_last_real(out, lengths)


# ── MODEL B — GRU (recurrent control) ────────────────────────────────────────

class GRUIntentEncoder(IntentEncoder):
    """MODEL B. Recurrent control.

    F-ARC-06: a naive GRU has NO padding-mask mechanism, so PAD embeddings would
    propagate through the recurrence and contaminate the final hidden state --
    handicapping Model B by a data-pipeline artefact and producing a spurious
    "Transformer wins". `pack_padded_sequence` makes the recurrence see ONLY real
    tokens, which is what makes the comparison fair."""

    NAME = "GRU"

    def __init__(self, hidden=None, n_layers=2, **kw):
        super().__init__(**kw)
        hidden = hidden or self.d_model
        self.hidden = hidden
        self.rnn = nn.GRU(self.d_model, hidden, num_layers=n_layers,
                          batch_first=True,
                          dropout=DROPOUT if n_layers > 1 else 0.0)
        self.proj = (nn.Identity() if hidden == self.d_model
                     else nn.Linear(hidden, self.d_model))
        self.norm = nn.LayerNorm(self.d_model)

    def _trunk(self, emb, lengths, pad_mask):
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hN = self.rnn(packed)              # hN: (layers, B, hidden)
        return self.norm(self.proj(hN[-1]))


# ── MODEL D — LSTM (second recurrent control) ────────────────────────────────

class LSTMIntentEncoder(IntentEncoder):
    """MODEL D. The other recurrent baseline.

    The v1 manuscript listed an LSTM comparison as future work (Sec. 6.4) and it
    stayed future work through v2-v6, so "attention vs recurrence" had until now
    been tested against ONE recurrent architecture. This closes that gap.

    Identical in every respect to GRUIntentEncoder except the cell: the same
    embedding, the same `pack_padded_sequence` padding discipline (F-ARC-06 --
    without it PAD embeddings contaminate the final hidden state and manufacture
    a spurious "Transformer wins"), the same projection and the same final
    LayerNorm. An LSTM cell has ~4/3 the parameters of a GRU cell at equal
    hidden width, so capacity is reported rather than assumed equal.
    """

    NAME = "LSTM"

    def __init__(self, hidden=None, n_layers=2, **kw):
        super().__init__(**kw)
        hidden = hidden or self.d_model
        self.hidden = hidden
        self.rnn = nn.LSTM(self.d_model, hidden, num_layers=n_layers,
                           batch_first=True,
                           dropout=DROPOUT if n_layers > 1 else 0.0)
        self.proj = (nn.Identity() if hidden == self.d_model
                     else nn.Linear(hidden, self.d_model))
        self.norm = nn.LayerNorm(self.d_model)

    def _trunk(self, emb, lengths, pad_mask):
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hN, _cN) = self.rnn(packed)       # hN: (layers, B, hidden)
        return self.norm(self.proj(hN[-1]))

# ── MODEL C — Set encoder (order-agnostic control) ───────────────────────────

class SetIntentEncoder(IntentEncoder):
    """MODEL C. Bag-of-techniques with masked mean pooling — ORDER-AGNOSTIC.

    This is the decisive control for the thesis's central claim. It is
    mathematically incapable of using sequence order (any permutation of the
    input yields the same h; asserted in test_permutation_invariance).
    If Model A does not beat Model C, order carries no exploitable signal in
    this corpus and the 'sequence modelling' framing must be withdrawn."""

    NAME = "SetEncoder"

    def __init__(self, hidden=512, **kw):
        super().__init__(**kw)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, self.d_model), nn.LayerNorm(self.d_model))

    def _trunk(self, emb, lengths, pad_mask):
        keep = (~pad_mask).unsqueeze(-1).float()           # (B, L, 1)
        pooled = (emb * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.mlp(pooled)


ENCODERS = {"Transformer": TransformerIntentEncoder,
            "GRU": GRUIntentEncoder,
            "SetEncoder": SetIntentEncoder}


# ── Capacity matching (fair comparison) ──────────────────────────────────────

def _n(params):
    return sum(p.numel() for p in params)


def match_capacity(target_trunk_params, tol=0.10, verbose=False):
    """Search each control's width so its TRUNK parameter count is within `tol`
    of Model A's. Without this, an A-vs-B result confounds architecture with
    capacity -- one of the commonest reviewer objections to encoder ablations."""
    out = {}
    best = None
    for hid in range(32, 200, 2):
        for nl in (1, 2, 3):
            m = GRUIntentEncoder(hidden=hid, n_layers=nl)
            d = abs(_n(m.trunk_parameters()) - target_trunk_params)
            if best is None or d < best[0]:
                best = (d, {"hidden": hid, "n_layers": nl},
                        _n(m.trunk_parameters()))
    out["GRU"] = {"kwargs": best[1], "params": best[2],
                  "rel": (best[2] - target_trunk_params) / target_trunk_params}
    best = None
    for hid in range(64, 1200, 8):
        m = SetIntentEncoder(hidden=hid)
        d = abs(_n(m.trunk_parameters()) - target_trunk_params)
        if best is None or d < best[0]:
            best = (d, {"hidden": hid}, _n(m.trunk_parameters()))
    out["SetEncoder"] = {"kwargs": best[1], "params": best[2],
                         "rel": (best[2] - target_trunk_params) / target_trunk_params}
    return out


def build_matched_suite():
    """Return {name: instantiated encoder} with controls capacity-matched to A."""
    a = TransformerIntentEncoder()
    tgt = _n(a.trunk_parameters())
    cfg = match_capacity(tgt)
    return ({"Transformer": a,
             "GRU": GRUIntentEncoder(**cfg["GRU"]["kwargs"]),
             "SetEncoder": SetIntentEncoder(**cfg["SetEncoder"]["kwargs"])},
            tgt, cfg)


# ── Parity / correctness tests ───────────────────────────────────────────────

def test_pad_invariance(enc, device="cpu", tol=1e-5):
    """F-MTH-07 — THE CLOCK-LEAK TEST.
    The same real prefix padded to different total lengths must give the SAME h.
    v1 (left-pad + absolute PE + fixed-index pooling) FAILS this by construction:
    that failure is the mechanism behind the measured h->step R^2 = 0.97."""
    enc.eval()
    real = [3, 11, 7, 2]
    outs = []
    with torch.no_grad():
        for total in (6, 10, MAX_SEQ_LEN):
            ids = torch.full((1, total), PAD_ID, dtype=torch.long)
            ids[0, :len(real)] = torch.tensor(real)
            outs.append(enc(ids.to(device),
                            torch.tensor([len(real)], device=device)).cpu().numpy())
    dev = max(float(np.abs(outs[0] - o).max()) for o in outs[1:])
    return dev, dev < tol


def test_permutation_invariance(enc, device="cpu"):
    """Model C must be permutation-INVARIANT (deviation ~0); A and B must be
    permutation-SENSITIVE (deviation > 0), else they are not using order."""
    enc.eval()
    a = torch.tensor([[3, 11, 7, 2]], dtype=torch.long, device=device)
    b = torch.tensor([[2, 7, 11, 3]], dtype=torch.long, device=device)
    ln = torch.tensor([4], device=device)
    with torch.no_grad():
        d = float((enc(a, ln) - enc(b, ln)).abs().max())
    return d


FLOAT32_EPS = 1e-6   # float32 accumulation noise; live dropout gives ~1e-1


def test_determinism(enc, device="cpu", n=10):
    """F-ARC-04 — h must be deterministic once frozen (dropout OFF).

    Tolerance is float32 epsilon, NOT exact zero: repeated float32 reductions
    differ in the last bits (~2e-7). Stochasticity from an accidentally-live
    dropout layer would show up ~5 orders of magnitude larger (~1e-1), so this
    threshold still catches the v1 defect it exists to catch."""
    enc.eval()
    ids = torch.tensor([[3, 11, 7, 2] + [PAD_ID] * (MAX_SEQ_LEN - 4)],
                       dtype=torch.long, device=device)
    ln = torch.tensor([4], device=device)
    with torch.no_grad():
        hs = np.stack([enc(ids, ln).cpu().numpy() for _ in range(n)])
    return float(hs.std(0).max())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(0)
    print("=" * 76)
    print("  TEMARL v2 — ENCODER SUITE (Model A / B / C)")
    print("=" * 76)

    suite, tgt, cfg = build_matched_suite()
    print(f"  vocab={VOCAB_SIZE}  d_model={D_MODEL}  MAX_SEQ_LEN={MAX_SEQ_LEN}")
    print(f"\n  [F-ARC-05] CAPACITY MATCHING (target = Model A trunk = {tgt:,})")
    print(f"  {'model':<12} {'trunk params':>13} {'rel':>8} {'total':>10}  config")
    for nm, m in suite.items():
        tp, tot = _n(m.trunk_parameters()), _n(m.parameters())
        rel = (tp - tgt) / tgt
        extra = cfg.get(nm, {}).get("kwargs", "n_layers=2,heads=4,d_ff=128")
        print(f"  {nm:<12} {tp:>13,} {rel:>+7.1%} {tot:>10,}  {extra}")
    ok_cap = all(abs((_n(m.trunk_parameters()) - tgt) / tgt) <= 0.10
                 for m in suite.values())
    print(f"  all within +-10%: {'PASS' if ok_cap else 'FAIL'}")

    print(f"\n  [PARITY] identical public interface")
    iface = ["forward", "encode", "pretrain_forward", "pretrain_head",
             "set_temperature", "calibrated_entropy", "get_h_numpy",
             "get_h_and_entropy_numpy", "freeze"]
    ok_if = all(hasattr(m, a) for m in suite.values() for a in iface)
    print(f"    {len(iface)} members present on all three: {'PASS' if ok_if else 'FAIL'}")

    ids = torch.randint(0, 20, (5, MAX_SEQ_LEN))
    ids[:, 6:] = PAD_ID
    ln = torch.full((5,), 6)
    for nm, m in suite.items():
        m.eval()
        with torch.no_grad():
            h, lg = m.pretrain_forward(ids, ln)
        hn = m.get_h_numpy(ids[0].numpy(), "cpu", lengths=[6])
        hh, ee = m.get_h_and_entropy_numpy(ids[0].numpy(), "cpu", lengths=[6])
        assert h.shape == (5, D_MODEL) and lg.shape == (5, VOCAB_SIZE)
        assert hn.shape == (D_MODEL,) and isinstance(ee, float)
    print(f"    shapes h=(B,64) logits=(B,{VOCAB_SIZE}), numpy helpers: PASS")

    print(f"\n  [F-MTH-07] PAD-INVARIANCE  (the clock-leak test; v1 fails by design)")
    all_pad_ok = True
    for nm, m in suite.items():
        dev, ok = test_pad_invariance(m)
        all_pad_ok &= ok
        print(f"    {nm:<12} max |h(pad=6) - h(pad=16)| = {dev:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  [CONTROL VALIDITY] permutation sensitivity")
    for nm, m in suite.items():
        d = test_permutation_invariance(m)
        want = "must be ~0 (order-blind)" if nm == "SetEncoder" else "must be >0 (uses order)"
        ok = (d < 1e-6) if nm == "SetEncoder" else (d > 1e-6)
        print(f"    {nm:<12} |h(seq) - h(perm)| = {d:.4e}   {want}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  [F-ARC-04] DETERMINISM after freeze()")
    for nm, m in suite.items():
        m.freeze()
        s = test_determinism(m)
        grad = any(p.requires_grad for p in m.parameters())
        ok = (s < FLOAT32_EPS) and not grad
        print(f"    {nm:<12} max std over 10 passes = {s:.2e} (< {FLOAT32_EPS:.0e})   "
              f"requires_grad={grad}   {'PASS' if ok else 'FAIL'}")
    print("=" * 76)
