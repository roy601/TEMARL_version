# -*- coding: utf-8 -*-
"""
TEMARL v4 — Rotary / no-position history encoders
==================================================
Canonical home for the positional-encoding variants that Phase 2.5 showed matter.

WHY THIS MODULE EXISTS. `encoders.py` builds its Transformer with ABSOLUTE
SINUSOIDAL position. Phase 2.5 measured that choice costing 2.6-7.0 points of
Top-1 on CAM-LDS, with the penalty growing with context length:

    context   Sin     RoPE    RoPE - Sin
      8      0.860   0.886      +0.026
     16      0.837   0.881      +0.044
     32      0.809   0.868      +0.059
     64      0.796   0.866      +0.070

Absolute position is also the design that produced the v1 clock leak, so there
were two independent reasons to suspect it. Dropping position entirely (NoPE)
also beats sinusoidal, which says the absolute encoding was actively harmful
rather than merely suboptimal.

`encoders.py` is NOT modified: it is a gated module and the sinusoidal variant
remains the historical baseline that every earlier result was measured on. These
classes are additive, and both are carried in comparisons so the change is a
measured variable rather than a silent substitution.

Rotary position: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary
Position Embedding", arXiv:2104.09864.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from encoders import DROPOUT, IntentEncoder, TransformerIntentEncoder


# ── rotary position ─────────────────────────────────────────────────────────

def _rope_cache(L, d, device):
    """(cos, sin) of shape (1, 1, L, d) for rotary embedding."""
    half = d // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    ang = torch.outer(torch.arange(L, device=device).float(), inv)   # (L, half)
    cos = torch.cos(ang).repeat_interleave(2, -1)                    # (L, d)
    sin = torch.sin(ang).repeat_interleave(2, -1)
    return cos[None, None], sin[None, None]


def _rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class _RoPELayer(nn.Module):
    """Pre-norm encoder layer with rotary position applied to q and k."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, L, D = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(h).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(h).view(B, L, self.h, self.dk).transpose(1, 2)
        cos, sin = _rope_cache(L, self.dk, x.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(pad_mask[:, None, None, :], -1e9)
        a = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.o(a))
        return x + self.drop(self.ff(self.n2(x)))


class TransformerRoPE(IntentEncoder):
    """Model A with ROTARY position. Phase 2.5's best history encoder."""

    NAME = "Transformer-RoPE"

    def __init__(self, n_layers=2, n_heads=4, d_ff=128, **kw):
        super().__init__(**kw)
        self.layers = nn.ModuleList([
            _RoPELayer(self.d_model, n_heads, d_ff, DROPOUT)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(self.d_model)

    def _trunk(self, emb, lengths, pad_mask):
        x = emb
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.gather_last_real(self.norm(x), lengths)


class TransformerNoPE(TransformerIntentEncoder):
    """Model A with NO positional encoding (ablation).

    Last-real-token pooling still exposes order through the data's causal
    structure, so this is a meaningful control rather than a guaranteed loss --
    and it beats the sinusoidal variant, which is the point.
    """

    NAME = "Transformer-NoPE"

    def _trunk(self, emb, lengths, pad_mask):
        out = self.enc(emb, src_key_padding_mask=pad_mask)
        return self.gather_last_real(out, lengths)


if __name__ == "__main__":
    import sys

    from vocab_v2 import PAD_ID

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 84)
    print("  TEMARL v4 — ROTARY / NO-POSITION ENCODER SELF-TEST")
    print("=" * 84)
    ok = True
    torch.manual_seed(0)
    B, L = 6, 16
    ids = torch.randint(0, 60, (B, L))
    lens = torch.randint(1, L + 1, (B,))
    for i in range(B):
        ids[i, lens[i]:] = PAD_ID

    for cls in (TransformerRoPE, TransformerNoPE):
        enc = cls()
        enc.eval()
        with torch.no_grad():
            h = enc(ids, lens)
        shape_ok = h.shape == (B, enc.d_model)
        finite = bool(torch.isfinite(h).all())
        print("  %-18s h=%s finite=%s  %s"
              % (cls.NAME, tuple(h.shape), finite,
                 "PASS" if shape_ok and finite else "FAIL"))
        ok &= shape_ok and finite

        # PAD-invariance: extra padding must not move h (the v1 clock-leak test)
        ids2 = torch.cat([ids, torch.full((B, 8), PAD_ID)], 1)
        with torch.no_grad():
            h2 = enc(ids2, lens)
        e = (h - h2).abs().max().item()
        print("      PAD-invariance max|d| = %.2e   %s"
              % (e, "PASS" if e < 1e-5 else "FAIL"))
        ok &= e < 1e-5

        # order sensitivity: h must change when the sequence is permuted
        perm = ids[:, torch.randperm(L)]
        with torch.no_grad():
            hp = enc(perm, lens)
        d = (h - hp).abs().max().item()
        print("      order sensitivity |h(seq)-h(perm)| = %.3e  %s"
              % (d, "PASS" if d > 1e-4 else "FAIL"))
        ok &= d > 1e-4

    print("=" * 84)
    print("  %s" % ("ALL PASS" if ok else "FAILURES PRESENT"))
    print("=" * 84)
    raise SystemExit(0 if ok else 1)
