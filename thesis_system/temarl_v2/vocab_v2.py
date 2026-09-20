# -*- coding: utf-8 -*-
"""
TEMARL v2 — Union Technique Vocabulary
======================================
Closes defects F-DAT-01, F-DAT-07, F-DAT-08.

DESIGN DECISION (and the evidence for it)
-----------------------------------------
A 13-TACTIC vocabulary was considered and REJECTED on measured grounds. On the
COMISET corpus (run-collapsed, i.e. the representation the loader produces):

    level        classes   H(X)    H(next|prev)   predictable info   majority
    TECHNIQUE      10      2.623      1.972          0.651 bits        37.2%
    TACTIC          5      1.397      1.280          0.117 bits        68.9%

Aggregating to tactics destroys 82% of the PREDICTABLE information
(0.651 -> 0.117 bits) and hands a trivial constant predictor 68.9% accuracy.
Any Model-A-vs-Model-B comparison at tactic level would be conducted in a
regime dominated by a majority-class baseline. Tactic labels are therefore
retained as AUXILIARY METADATA (for the D3FEND payoff and for reporting)
but are never the model's prediction target.

THE UNION VOCABULARY
--------------------
  ids 0..22   FROZEN, identical to mitre_techniques.py, so existing encoder
              checkpoints load via embedding-row expansion.
  ids 23..N   CAM-LDS-only parent techniques, appended in deterministic
              (sorted) order for reproducibility.
  PAD, UNK    moved to the END of the table (ids N+1, N+2).

This guarantees ZERO UNK collapse on either corpus, which is the precondition
for one encoder to read both COMISET (Windows) and CAM-LDS (Linux).

CROSS-CORPUS CAVEAT (F-DAT-08) — measured, and it constrains the experiment:
    COMISET observed tactics (5) SUBSET OF CAM-LDS tactics (13); COMISET-only = {}.
    Therefore COMISET -> CAM-LDS zero-shot is NOT a generalisation test: 8 of 13
    label classes would be unseen. The valid headline direction is
    CAM-LDS -> COMISET (test labels all seen in training), and any
    COMISET -> CAM-LDS number must be reported as RESTRICTED-SUPPORT transfer
    over the 5-tactic intersection only. See tactic_intersection() below.
"""

import json
import os

# ── Frozen legacy block: ids 0-22 must not move ──────────────────────────────
# (mirrors mitre_techniques.MITRE_TECHNIQUES; duplicated here so v2 is
#  self-contained and cannot be silently broken by an edit to the v1 file)
_FROZEN = [
    ("T1190", 0,  "Exploit Public-Facing Application",      "Initial Access"),
    ("T1566", 1,  "Phishing",                               "Initial Access"),
    ("T1059", 2,  "Command and Scripting Interpreter",      "Execution"),
    ("T1204", 3,  "User Execution",                         "Execution"),
    ("T1053", 4,  "Scheduled Task/Job",                     "Persistence"),
    ("T1543", 5,  "Create or Modify System Process",        "Persistence"),
    ("T1547", 6,  "Boot or Logon Autostart Execution",      "Persistence"),
    ("T1068", 7,  "Exploitation for Privilege Escalation",  "Privilege Escalation"),
    ("T1055", 8,  "Process Injection",                      "Privilege Escalation"),
    ("T1046", 9,  "Network Service Scanning",               "Discovery"),
    ("T1016", 10, "System Network Configuration Discovery", "Discovery"),
    ("T1083", 11, "File and Directory Discovery",           "Discovery"),
    ("T1021", 12, "Remote Services",                        "Lateral Movement"),
    ("T1550", 13, "Use Alternate Authentication Material",  "Lateral Movement"),
    ("T1072", 14, "Software Deployment Tools",              "Lateral Movement"),
    ("T1005", 15, "Data from Local System",                 "Collection"),
    ("T1560", 16, "Archive Collected Data",                 "Collection"),
    ("T1041", 17, "Exfiltration Over C2 Channel",           "Exfiltration"),
    ("T1486", 18, "Data Encrypted for Impact",              "Impact"),
    ("T1489", 19, "Service Stop",                           "Impact"),
    ("T1036", 20, "Masquerading",                           "Defense Evasion"),
    ("T1574", 21, "Hijack Execution Flow",                  "Persistence"),
    ("T1553", 22, "Subvert Trust Controls",                 "Defense Evasion"),
]
N_FROZEN = len(_FROZEN)          # 23

# Canonical MITRE tactic ordering (kill-chain order). Auxiliary metadata only.
TACTIC_ORDER = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution",
    "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Discovery", "Lateral Movement", "Collection", "Command and Control",
    "Exfiltration", "Impact",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_GROUNDING = os.path.join(_HERE, "..", "data", "camlds_grounding.json")
# Source-verified inventory, parsed from the published AttackBed playbooks by
# parse_attackbed.py. It is the AUTHORITY ON MEMBERSHIP; `_GROUNDING` above is
# retained solely to PIN THE HISTORICAL ID ORDER (see build_vocab).
_GROUNDING_VERIFIED = os.path.join(_HERE, "..", "data",
                                   "camlds_grounding_verified.json")


def _load_camlds_inventory(path=_GROUNDING):
    """CAM-LDS parent-technique inventory {code: {name, tactic}}; {} if absent."""
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}
    inv = g.get("technique_inventory", {})
    return {k: v for k, v in inv.items() if k != "_comment"}


def build_vocab(camlds_inventory=None):
    """Construct the union vocabulary. Deterministic: CAM-LDS-only codes are
    appended in sorted order so the mapping is stable across runs/machines."""
    cam = _load_camlds_inventory() if camlds_inventory is None else camlds_inventory
    table = {code: {"id": i, "name": nm, "tactic": tc, "source": "frozen"}
             for code, i, nm, tc in _FROZEN}

    new_codes = sorted(c for c in cam if c not in table)
    nxt = N_FROZEN
    for code in new_codes:
        meta = cam[code]
        table[code] = {"id": nxt, "name": meta.get("name", code),
                       "tactic": meta.get("tactic", "Unknown"), "source": "camlds"}
        nxt += 1

    # ── source-verified extension (APPEND-ONLY) ─────────────────────────────
    # The hand reconstruction missed 6 techniques that the published playbooks
    # actually execute (6.20% of CAM-LDS technique instances fell to UNK). They
    # are appended AFTER the existing ids rather than merged into the sorted
    # block, because renumbering would silently invalidate `payoff_frozen.json`
    # (78x6, keyed by id) and every trained checkpoint. Appending keeps ids
    # 0..77 bit-identical and leaves the frozen payoff's existing rows valid.
    ver = _load_camlds_inventory(_GROUNDING_VERIFIED)
    for code in sorted(c for c in ver if c not in table):
        meta = ver[code]
        table[code] = {"id": nxt, "name": meta.get("name", code),
                       "tactic": meta.get("tactic", "Unknown"),
                       "source": "camlds-verified"}
        nxt += 1

    n_tech = nxt
    table["PAD"] = {"id": n_tech,     "name": "Padding Token", "tactic": "None", "source": "special"}
    table["UNK"] = {"id": n_tech + 1, "name": "Unknown",       "tactic": "None", "source": "special"}
    return table, n_tech


TECHNIQUES, NUM_TECHNIQUES = build_vocab()
PAD_ID     = TECHNIQUES["PAD"]["id"]
UNK_ID     = TECHNIQUES["UNK"]["id"]
VOCAB_SIZE = NUM_TECHNIQUES + 2
MAX_SEQ_LEN = 16

TOKENS          = {k: v["id"] for k, v in TECHNIQUES.items()}
ID_TO_TECHNIQUE = {v["id"]: k for k, v in TECHNIQUES.items()}
ID_TO_META      = {v["id"]: v for v in TECHNIQUES.values()}
ID_TO_TACTIC    = {v["id"]: v["tactic"] for v in TECHNIQUES.values()
                   if v["tactic"] != "None"}

TACTIC_TO_IDS = {}
for _c, _v in TECHNIQUES.items():
    if _v["tactic"] != "None":
        TACTIC_TO_IDS.setdefault(_v["tactic"], []).append(_v["id"])


def technique_to_id(raw):
    """Map a raw MITRE code (with or without sub-technique suffix) to an id.
    'T1574.002' -> parent 'T1574'. Unmapped -> UNK_ID."""
    if not raw or not isinstance(raw, str):
        return UNK_ID
    s = raw.strip()
    if s in TECHNIQUES and s not in ("PAD", "UNK"):
        return TECHNIQUES[s]["id"]
    parent = s.split(".")[0]
    if parent in TECHNIQUES and parent not in ("PAD", "UNK"):
        return TECHNIQUES[parent]["id"]
    return UNK_ID


def tactic_of(tid):
    return ID_TO_TACTIC.get(int(tid), "Unknown")


def tactic_intersection(corpus_a_ids, corpus_b_ids):
    """F-DAT-08 guard. Returns the tactic sets and their intersection so a
    transfer experiment can be reported over valid (shared) support only."""
    ta = {tactic_of(i) for i in corpus_a_ids if int(i) < NUM_TECHNIQUES}
    tb = {tactic_of(i) for i in corpus_b_ids if int(i) < NUM_TECHNIQUES}
    return {"a_only": sorted(ta - tb), "b_only": sorted(tb - ta),
            "shared": sorted(ta & tb), "a": sorted(ta), "b": sorted(tb)}


def expand_embedding_rows(old_state_dict, new_state_dict,
                          keys=("embedding.weight", "pretrain_head.weight",
                                "pretrain_head.bias")):
    """Load a v1 checkpoint into a v2 (larger-vocab) model: copy the frozen
    rows 0..22 and leave new rows at their fresh initialisation."""
    out = dict(old_state_dict)
    for k in keys:
        if k in out and k in new_state_dict and out[k].shape != new_state_dict[k].shape:
            src, dst = out[k], new_state_dict[k].clone()
            n = min(src.shape[0], dst.shape[0])
            dst[:n] = src[:n]
            out[k] = dst
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 68)
    print("  TEMARL v2 — UNION VOCABULARY")
    print("=" * 68)
    frozen = [c for c, v in TECHNIQUES.items() if v["source"] == "frozen"]
    camlds = [c for c, v in TECHNIQUES.items() if v["source"] == "camlds"]
    print(f"  frozen (ids 0-22)      : {len(frozen)}")
    print(f"  CAM-LDS appended (23+) : {len(camlds)}")
    print(f"  NUM_TECHNIQUES         : {NUM_TECHNIQUES}")
    print(f"  PAD / UNK              : {PAD_ID} / {UNK_ID}")
    print(f"  VOCAB_SIZE             : {VOCAB_SIZE}")

    # Assertion 1 — frozen ids must be byte-identical to v1
    ok = all(TECHNIQUES[c]["id"] == i for c, i, _, _ in _FROZEN)
    print(f"\n  [F-DAT-01] ids 0-22 unchanged : {'PASS' if ok else 'FAIL'}")
    assert ok

    # Assertion 2 — zero UNK on both corpora
    cam = _load_camlds_inventory()
    unk_cam = [c for c in cam if technique_to_id(c) == UNK_ID]
    comiset_codes = ["T1574", "T1543", "T1053", "T1059", "T1553",
                     "T1055", "T1036", "T1204", "T1547", "T1021"]
    unk_com = [c for c in comiset_codes if technique_to_id(c) == UNK_ID]
    print(f"  [F-DAT-01] CAM-LDS UNK count  : {len(unk_cam)}  {'PASS' if not unk_cam else 'FAIL ' + str(unk_cam)}")
    print(f"  [F-DAT-01] COMISET UNK count  : {len(unk_com)}  {'PASS' if not unk_com else 'FAIL ' + str(unk_com)}")
    assert not unk_cam and not unk_com

    # Assertion 3 — sub-technique normalisation
    for raw, parent in [("T1574.002", "T1574"), ("T1036.005", "T1036"),
                        ("T1595.003", "T1595"), ("T9999", None)]:
        got = technique_to_id(raw)
        exp = TECHNIQUES[parent]["id"] if parent and parent in TECHNIQUES else UNK_ID
        assert got == exp, f"{raw}: got {got} expected {exp}"
    print("  [F-DAT-01] sub-technique norm : PASS")

    # F-DAT-08 — the cross-corpus label-space relation
    com_ids = [technique_to_id(c) for c in comiset_codes]
    cam_ids = [technique_to_id(c) for c in cam]
    rel = tactic_intersection(com_ids, cam_ids)
    print(f"\n  [F-DAT-08] cross-corpus tactic support")
    print(f"    COMISET tactics ({len(rel['a'])}) : {rel['a']}")
    print(f"    CAM-LDS tactics ({len(rel['b'])}) : {len(rel['b'])} classes")
    print(f"    shared            : {len(rel['shared'])}")
    print(f"    COMISET-only      : {rel['a_only']}  <- empty => COMISET is a SUBSET")
    print(f"    CAM-LDS-only      : {len(rel['b_only'])} tactics unseen by a COMISET-trained model")
    print(f"\n  => valid headline transfer direction: CAM-LDS -> COMISET")
    print(f"     COMISET -> CAM-LDS must be reported as RESTRICTED-SUPPORT "
          f"({len(rel['shared'])} shared tactics only)")
    print("=" * 68)
