# -*- coding: utf-8 -*-
"""
TEMARL — device selection
==========================
Every v2/v3/v4 script chose its device with

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

which silently ignores Intel XPU (Arc) and AMD ROCm builds. On the Arc machine
`torch.cuda.is_available()` is False but `torch.xpu.is_available()` is True, so
the GPU sat at ~9% while the CPU ran at 90%.

BUT MEASURE BEFORE MOVING. Benchmarked on this machine (torch 2.8.0+xpu, Arc A-series,
TransformerRoPE 4 layers / 8 heads / d_ff 512 / d_model 32, context 8):

    batch    CPU ms/step   XPU ms/step   speedup
        4         13.68         25.62       0.53x   <- RL rollout (4 agents)
       32         20.33         25.19       0.81x
      128         39.81         25.86       1.54x   <- encoder pretraining
      512         95.65         27.01       3.54x

**The GPU is SLOWER for small batches**, because kernel-launch latency dominates.
The entity RL loop runs one forward pass over 4 agents per environment step, so
it is firmly in the regime where CPU wins. Moving the whole v4 experiment to XPU
would make it slower (~150s vs ~118s per arm), not faster.

So the default stays CPU-unless-CUDA, and XPU is opt-in per workload:

    DEVICE  = pick_device()                 # cuda > cpu; XPU only if asked
    PRETRAIN_DEVICE = pick_device(prefer_xpu=True)   # worth it at batch >= 128

Set TEMARL_DEVICE=cpu|cuda|xpu to force a choice.
"""

from __future__ import annotations

import os

import torch

# Batch size above which the accelerator's throughput beats its launch latency,
# from the benchmark above. Below this, CPU is faster.
XPU_BATCH_BREAKEVEN = 128


def available_devices() -> dict:
    d = {"cpu": True, "cuda": torch.cuda.is_available(), "xpu": False}
    if hasattr(torch, "xpu"):
        try:
            d["xpu"] = bool(torch.xpu.is_available())
        except Exception:
            d["xpu"] = False
    return d


def pick_device(prefer_xpu: bool = False, batch_size: int | None = None) -> str:
    """Choose a device.

    CUDA is preferred whenever present (it is fast at every batch size we use).
    XPU is selected only when explicitly requested, or when `batch_size` is at
    or above the measured break-even point.
    """
    forced = os.environ.get("TEMARL_DEVICE")
    if forced:
        return forced
    avail = available_devices()
    if avail["cuda"]:
        return "cuda"
    if avail["xpu"] and (prefer_xpu or
                         (batch_size is not None
                          and batch_size >= XPU_BATCH_BREAKEVEN)):
        return "xpu"
    return "cpu"


def sync(device: str) -> None:
    """Barrier, so timings are honest on an asynchronous backend."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def describe() -> str:
    a = available_devices()
    parts = ["cpu(%d threads)" % torch.get_num_threads()]
    if a["cuda"]:
        parts.append("cuda:%s" % torch.cuda.get_device_name(0))
    if a["xpu"]:
        try:
            parts.append("xpu:%s" % torch.xpu.get_device_name(0))
        except Exception:
            parts.append("xpu")
    return " | ".join(parts)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 76)
    print("  TEMARL — DEVICE SELECTION")
    print("=" * 76)
    print("  torch            : %s (cuda build: %s)"
          % (torch.__version__, torch.version.cuda))
    print("  available        : %s" % available_devices())
    print("  hardware         : %s" % describe())
    print()
    print("  pick_device()                       -> %s" % pick_device())
    print("  pick_device(prefer_xpu=True)        -> %s"
          % pick_device(prefer_xpu=True))
    print("  pick_device(batch_size=4)           -> %s" % pick_device(batch_size=4))
    print("  pick_device(batch_size=128)         -> %s"
          % pick_device(batch_size=128))
    print()
    print("  NOTE: XPU is ~2x SLOWER than CPU at batch 4 (the RL rollout size).")
    print("        Do not move a whole entity-RL experiment onto it.")
    print("=" * 76)
