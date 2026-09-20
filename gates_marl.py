# -*- coding: utf-8 -*-
"""
TEMARL v7 — validity gates (pre-registered thresholds)
======================================================
The thresholds below were written on 2026-09-18, BEFORE any ladder number had
been computed, and are not to be changed after seeing one. Every gate is a
function of the architecture-free reference ladder (`baselines_marl.py`), so
no gate can be passed or failed by a choice of history encoder.

  G1 DISCRIMINATION  null < random < best-fixed < ceiling, and the headroom
                     (ceiling - best fixed) is at least 1 dwell step AND at
                     least 25% of the ceiling. Without headroom no learned
                     team can be distinguished from a constant.
  G2 COORDINATION    THE MARL TEST. (i) The coordinated clairvoyant planner
                     beats the better of the two uncoordinated clairvoyant
                     variants by >= 15% of the ceiling; (ii) laying a
                     breadcrumb WITH a partner beats laying it without one
                     (paired 95% CI above 0); (iii) exceeding the capacity
                     hurts (paired 95% CI below 0). If G2 fails, the
                     environment does not require coordination and is not a
                     MARL problem.
  G3 INTENT VALUE    knowing the hidden script (coordinated) beats knowing
                     nothing about it by >= 10% of the ceiling. Without it,
                     no history encoder can matter.
  G4 SEQUENCE VALUE  knowing the script AND the current technique beats
                     knowing only the script by >= 3% of the ceiling. If G4
                     fails, the architecture contrasts (Transformer vs
                     GRU/LSTM/Set) are declared UNINTERPRETABLE in advance, as
                     v4 taught: an order-blind encoder could then match any
                     sequence model by construction.
  G5 LOSO            for every held-out script, the ceiling beats the best
                     fixed joint action chosen on the six training scripts
                     (paired 95% CI above 0), so generalisation is measurable.

The ceiling is the coordinated clairvoyant planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

import _frozen
from baselines_marl import (Beliefs, Fixed, Planner, best_fixed, evaluate,
                            ladder, paired_diff, summarise, tune_theta)
from env_marl import SCRIPT_IDS, EnvConfig, EpisodeSource
from scripts_marl import loso_folds

THRESHOLDS = {
    "G1_headroom_abs": 1.0,
    "G1_headroom_rel": 0.25,
    "G2_coordination_rel": 0.15,
    "G3_intent_rel": 0.10,
    "G4_sequence_rel": 0.03,
    "G5_ci_lower_gt": 0.0,
}

CEIL = "clairvoyant/coord"
HERE = os.path.dirname(os.path.abspath(__file__))


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()[:16]


def gates_g1_g4(lad: dict) -> dict:
    S, R = lad["summary"], lad["rows"]
    ceil = S[CEIL]["dwell"]
    th = THRESHOLDS
    g = {}

    order = [S[k]["dwell"] for k in ("null", "random", "best_fixed", CEIL)]
    head = ceil - S["best_fixed"]["dwell"]
    g["G1"] = {
        "ladder": dict(zip(("null", "random", "best_fixed", "ceiling"), order)),
        "headroom": head, "headroom_rel": head / ceil if ceil > 0 else 0.0,
        "pass": bool(all(a < b for a, b in zip(order, order[1:]))
                     and head >= th["G1_headroom_abs"]
                     and head >= th["G1_headroom_rel"] * ceil)}

    unc = max(("clairvoyant/uncoord-naive", "clairvoyant/uncoord-averse"),
              key=lambda k: S[k]["dwell"])
    coord_val = ceil - S[unc]["dwell"]
    comp, comp_ci = paired_diff(R[CEIL], R["clairvoyant/coord-nopartner"])
    cap, cap_ci = paired_diff(R["clairvoyant/coord-fillall"], R[CEIL])
    g["G2"] = {
        "best_uncoordinated": unc, "coordination_value": coord_val,
        "coordination_rel": coord_val / ceil if ceil > 0 else 0.0,
        "complementarity": comp, "complementarity_ci": comp_ci,
        "over_capacity": cap, "over_capacity_ci": cap_ci,
        "pass": bool(coord_val >= th["G2_coordination_rel"] * ceil
                     and comp - comp_ci > 0 and cap + cap_ci < 0)}

    iv = S["script/coord"]["dwell"] - S["nointent/coord"]["dwell"]
    g["G3"] = {"intent_value": iv, "intent_rel": iv / ceil if ceil > 0 else 0.0,
               "pass": bool(iv >= th["G3_intent_rel"] * ceil)}

    sv = S["transition/coord"]["dwell"] - S["script/coord"]["dwell"]
    g["G4"] = {"sequence_value": sv, "sequence_rel": sv / ceil if ceil > 0 else 0.0,
               "pass": bool(sv >= th["G4_sequence_rel"] * ceil)}
    return g


def gate_g5(cfg: EnvConfig, n_sel=300, n_fixed_sel=120, n_eval=500, seed=0):
    """Leave-one-script-out: is generalisation measurable for every script?"""
    out = {}
    for fold in loso_folds():
        B = Beliefs(fold["train"])
        sel = EpisodeSource(40_000 + seed, fold["train"]).take(n_sel)
        ev = EpisodeSource(50_000 + seed, [fold["held_out"]]).take(n_eval)
        bf, _ = best_fixed(sel[:n_fixed_sel], cfg)
        th = tune_theta(lambda t: Planner("clairvoyant", True, B, t), sel, cfg)
        r_ceil = evaluate(Planner("clairvoyant", True, B, th), ev, cfg)
        r_bf = evaluate(Fixed(bf), ev, cfg)
        d, ci = paired_diff(r_ceil, r_bf)
        out[fold["held_out"]] = {
            "ceiling": summarise(r_ceil)["dwell"],
            "best_fixed_from_train": summarise(r_bf)["dwell"],
            "best_fixed_joint": bf, "diff": d, "ci": ci,
            "pass": bool(d - ci > THRESHOLDS["G5_ci_lower_gt"])}
    return {"folds": out, "pass": all(v["pass"] for v in out.values())}


def print_gates(g: dict):
    g1, g2, g3, g4 = g["G1"], g["G2"], g["G3"], g["G4"]
    L = g1["ladder"]
    print("  G1 discrimination  %s  null %.3f < random %.3f < best-fixed %.3f "
          "< ceiling %.3f ; headroom %.3f (%.0f%%)"
          % ("PASS" if g1["pass"] else "FAIL", L["null"], L["random"],
             L["best_fixed"], L["ceiling"], g1["headroom"], 100 * g1["headroom_rel"]))
    print("  G2 coordination    %s  coord value %.3f (%.0f%% of ceiling, vs %s); "
          "breadcrumb partner %+.3f +- %.3f; over capacity %+.3f +- %.3f"
          % ("PASS" if g2["pass"] else "FAIL", g2["coordination_value"],
             100 * g2["coordination_rel"], g2["best_uncoordinated"],
             g2["complementarity"], g2["complementarity_ci"],
             g2["over_capacity"], g2["over_capacity_ci"]))
    print("  G3 intent value    %s  %.3f (%.0f%% of ceiling)"
          % ("PASS" if g3["pass"] else "FAIL", g3["intent_value"],
             100 * g3["intent_rel"]))
    print("  G4 sequence value  %s  %.3f (%.1f%% of ceiling)"
          % ("PASS" if g4["pass"] else "FAIL", g4["sequence_value"],
             100 * g4["sequence_rel"]))
    if "G5" in g:
        for sid, v in g["G5"]["folds"].items():
            print("  G5 LOSO %-3s       %s  ceiling %.3f vs best-fixed(train) %.3f "
                  "diff %+.3f +- %.3f" % (sid, "PASS" if v["pass"] else "FAIL",
                                          v["ceiling"], v["best_fixed_from_train"],
                                          v["diff"], v["ci"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "env_config.json"))
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1,
                    help="final gates use a stream disjoint from calibration (0)")
    ap.add_argument("--out", default=os.path.join(HERE, "results_marl", "gates.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg_d = json.load(open(args.config, encoding="utf-8"))["config"]
    cfg = EnvConfig(**cfg_d)
    t0 = time.time()
    print("=" * 110)
    print("  TEMARL v7 — VALIDITY GATES (thresholds fixed in source before any "
          "ladder number existed)")
    print("  config %s | gates file %s | frozen %s"
          % (cfg_d, file_sha(__file__), _frozen.fingerprint()))
    print("=" * 110)
    lad = ladder(cfg, SCRIPT_IDS, n_eval=args.n_eval, seed=args.seed, verbose=True)
    g = gates_g1_g4(lad)
    g["G5"] = gate_g5(cfg, seed=args.seed)
    print()
    print_gates(g)
    allpass = all(g[k]["pass"] for k in ("G1", "G2", "G3", "G4", "G5"))
    print("\n  ALL GATES: %s   (%.0f s)" % ("PASS" if allpass else "FAIL",
                                           time.time() - t0))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"config": cfg_d, "thresholds": THRESHOLDS, "gates": g,
               "summary": lad["summary"], "meta": lad["meta"],
               "gates_sha": file_sha(__file__),
               "frozen": _frozen.fingerprint(), "all_pass": allpass},
              open(args.out, "w", encoding="utf-8"), indent=1, default=float)
    print("  wrote", os.path.relpath(args.out, HERE))
    print("=" * 110)


if __name__ == "__main__":
    main()
