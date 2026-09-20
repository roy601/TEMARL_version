"""Paper-based ATT&CK sequence encoders. See ARCHITECTURE_AND_AUDIT.md."""
from functools import lru_cache

import torch
from torch import nn

import _frozen
from encoders import IntentEncoder, TransformerIntentEncoder
from vocab_v2 import PAD_ID

H_DIM = 64
ARMS = ("Transformer", "GRU", "LSTM", "NoHistory")


class ChoGRUCell(nn.Module):
    """Cho et al. (2014), equations 5-8: U(r * h), not r * U(h)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x = nn.Linear(input_size, 3 * hidden_size)
        self.h_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.h_candidate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        xr, xz, xn = self.x(x).chunk(3, -1)
        hr, hz = self.h_gates(h).chunk(2, -1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        candidate = torch.tanh(xn + self.h_candidate(r * h))
        return z * h + (1 - z) * candidate


class LSTM1997Cell(nn.Module):
    """1997 constant-error-carousel cell, one cell per block, no forget gate.

    Section 4: c'=c+i*g(x,h), h'=o*h(c'). The paper's scaled sigmoid
    choices are g(x)=2*tanh(x/2), h(c)=tanh(c/2). Training uses modern BPTT.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x = nn.Linear(input_size, 3 * hidden_size)
        self.h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        i, o, g = (self.x(x) + self.h(h)).chunk(3, -1)
        c = c + torch.sigmoid(i) * (2 * torch.tanh(g / 2))
        h = torch.sigmoid(o) * torch.tanh(c / 2)
        return h, c


class ValidLengths:
    def forward(self, token_ids, lengths=None):
        if token_ids.ndim == 1:
            token_ids = token_ids[None]
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.max_len:
            raise ValueError("Expected nonempty right-padded windows within max_len")
        lengths = ((token_ids != PAD_ID).sum(1) if lengths is None else
                   torch.as_tensor(lengths, device=token_ids.device).long())
        if lengths.shape != (len(token_ids),) or (lengths < 0).any() or (lengths > token_ids.shape[1]).any():
            raise ValueError("Invalid sequence lengths")
        tail = torch.arange(token_ids.shape[1], device=token_ids.device)[None] >= lengths[:, None]
        clean = token_ids.masked_fill(tail, PAD_ID)
        if ((clean == PAD_ID) & ~tail).any():
            raise ValueError("PAD inside real prefix")
        result = super().forward(clean, lengths)
        return result.masked_fill((lengths == 0)[:, None], 0)


class Transformer(ValidLengths, TransformerIntentEncoder):
    """FlowTransformer-informed shallow encoder with last-real-token pooling."""


class Recurrent(ValidLengths, IntentEncoder):
    def __init__(self, kind, hidden=100, n_layers=2):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        cell = ChoGRUCell if kind == "GRU" else LSTM1997Cell
        self.cells = nn.ModuleList(cell(H_DIM if i == 0 else hidden, hidden)
                                   for i in range(n_layers))
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Identity() if hidden == H_DIM else nn.Linear(hidden, H_DIM)
        self.norm = nn.LayerNorm(H_DIM)

    def _trunk(self, emb, lengths, pad_mask):
        x = emb
        for layer, cell in enumerate(self.cells):
            h = x.new_zeros(len(x), self.hidden)
            c = torch.zeros_like(h)
            outputs = []
            for t in range(x.shape[1]):
                live = (t < lengths)[:, None]
                if self.kind == "GRU":
                    h = torch.where(live, cell(x[:, t], h), h)
                else:
                    hn, cn = cell(x[:, t], (h, c))
                    h, c = torch.where(live, hn, h), torch.where(live, cn, c)
                outputs.append(h)
            x = torch.stack(outputs, 1)
            if layer < len(self.cells) - 1:
                x = self.drop(x)
        return self.norm(self.proj(h))


@lru_cache(maxsize=1)
def arm_specs():
    # fork_rng prevents capacity search from perturbing model initialization.
    with torch.random.fork_rng(devices=[]):
        target = sum(p.numel() for p in Transformer().trunk_parameters())
    best = None
    for hidden in range(32, 257):
        for layers in (1, 2):
            p = sum(3 * hidden * ((H_DIM if i == 0 else hidden) + hidden + 1)
                    for i in range(layers))
            p += 0 if hidden == H_DIM else hidden * H_DIM + H_DIM
            p += 2 * H_DIM
            candidate = (abs(p - target), hidden, layers, p)
            if best is None or candidate < best:
                best = candidate
    _, hidden, layers, count = best
    if abs(count - target) / target > 0.1:
        raise RuntimeError("Recurrent capacity matching failed")
    return {"Transformer": {"kwargs": {}, "params": target},
            "GRU": {"kwargs": {"hidden": hidden, "n_layers": layers}, "params": count},
            "LSTM": {"kwargs": {"hidden": hidden, "n_layers": layers}, "params": count},
            "NoHistory": {"kwargs": {}, "params": 0}}


def build_encoder(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown encoder: {arm}")
    if arm == "NoHistory":
        return None
    return Transformer() if arm == "Transformer" else Recurrent(arm, **arm_specs()[arm]["kwargs"])
