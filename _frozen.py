# -*- coding: utf-8 -*-
"""
TEMARL v7 — access to the frozen TEMARL v2 modules, with integrity checks
=========================================================================
v7 is an ADDITIVE experiment. It reuses, without modification, the frozen
D3FEND payoff, the 84-technique vocabulary, the v5 chain estimator, the history
encoders and the GMU trunk from `thesis_system/temarl_v2/`. This portable bundle
includes those dependencies locally. This module puts that directory on
`sys.path` and verifies that every reused file is byte-identical (modulo line
endings) to the version v7 was designed against.

If any check fails, v7 refuses to run. A frozen input changing under an
experiment is exactly the kind of silent confound this project exists to rule
out, so the failure is loud and must be resolved by a human, never bypassed.

Line endings are normalised (CRLF -> LF) before hashing because git on Windows
checks files out with CRLF (`core.autocrlf=true` on the development machine),
and the GPU machine may differ. Content, not line endings, is what is frozen.

No absolute path appears here: everything is resolved relative to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE  # Standalone bundle: frozen dependencies live inside this folder.
V2_DIR = os.path.join(REPO, "thesis_system", "temarl_v2")
DATA_DIR = os.path.join(REPO, "thesis_system", "data")

if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

# sha256 of each reused file, CRLF normalised to LF, recorded 2026-09-18 when
# v7 was designed. Paths are relative to the repository root.
FROZEN_SHA256 = {
    "thesis_system/data/camlds_grounding_verified.json":
        "20006361c0e6b6a46d9da02e67606cdaa0c3e375a4a0c7396c24484d1b88d7be",
    "thesis_system/temarl_v2/payoff_frozen.json":
        "854ebf25a123e5b471e9571aa2789c5b94d1457b054f2f93acb564232d508f97",
    "thesis_system/temarl_v2/d3fend_payoff.py":
        "ca85718fb88107338d9bf522e947a84b57ef03b437c7b6796df1cfae6621597d",
    "thesis_system/temarl_v2/vocab_v2.py":
        "b57d0e1b21153cfd24ba48969b3a3af071d747b68aaa3218f7a8c7cb60d740d0",
    "thesis_system/temarl_v2/profiles_v5.py":
        "1b0dab40c2545a8b63305f61485666c14244ae2e39ccd1584e84b8befa379fd8",
    "thesis_system/temarl_v2/env_v2.py":
        "af9e9d61b67e8564e028faa5b94bad2c06f0a0c393260922ed5b60b184e961fe",
    "thesis_system/temarl_v2/encoders.py":
        "f0c0603229a8b4a0ca170670bb9234c7b1e5796d0358d8ec8ae759f8f690a374",
    "thesis_system/temarl_v2/encoders_rope.py":
        "8a9f036948eb182a3b67bf4ff30479b0b6105e0466e3e913f1ee5b07643be9f2",
    "thesis_system/temarl_v2/policy.py":
        "b77aa6fa5be82373e55084ae79181b07ab03677508974df72cb6d297128f4c57",
    "thesis_system/temarl_v2/device_util.py":
        "e6c867784e9f3c87b3975424055225e23b6d85d262129c1a7eaa526f13f73c1a",
    "thesis_system/data/camlds_grounding.json":
        "e8b7bbc110ade5f2e5ac92ad913940263355f2c042f1022c4880bdaf160ff569",
}


def normalised_sha256(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify(verbose: bool = False) -> dict:
    """Check every frozen input. Raises RuntimeError on the first mismatch."""
    report = {}
    for rel, want in FROZEN_SHA256.items():
        path = os.path.join(REPO, *rel.split("/"))
        if not os.path.exists(path):
            raise RuntimeError(f"frozen input missing: {rel}")
        have = normalised_sha256(path)
        report[rel] = have == want
        if have != want:
            raise RuntimeError(
                f"frozen input CHANGED: {rel}\n  expected {want}\n  found    {have}\n"
                "v7 was designed against the expected version. Do not edit the "
                "hash to make this pass; find out why the file changed.")

    # The payoff that the code builds must equal the frozen JSON, cell for cell.
    import numpy as np
    from d3fend_payoff import PAYOFF
    with open(os.path.join(V2_DIR, "payoff_frozen.json"), encoding="utf-8") as f:
        fz = json.load(f)
    A = np.asarray(fz["payoff_matrix"], dtype=np.float64)
    if A.shape != PAYOFF.shape or not np.array_equal(A, PAYOFF):
        raise RuntimeError("PAYOFF built by d3fend_payoff.py differs from "
                           "payoff_frozen.json")
    report["payoff_matrix_equal"] = True
    if verbose:
        for k, v in report.items():
            print("  %-48s %s" % (k, "OK" if v else "MISMATCH"))
    return report


def fingerprint() -> str:
    """One short id for the full set of frozen inputs (stored with results)."""
    h = hashlib.sha256()
    for rel in sorted(FROZEN_SHA256):
        h.update(rel.encode()); h.update(FROZEN_SHA256[rel].encode())
    return h.hexdigest()[:16]


# Verify on import: nothing in v7 may run against a drifted frozen input.
verify()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("frozen inputs (v2 dir: %s)" % os.path.relpath(V2_DIR, HERE))
    verify(verbose=True)
    print("fingerprint:", fingerprint())
