#!/usr/bin/env python
"""Compare g4numi charged-pion production off the NuMI target with the MIPP
measurement of Paley et al., Phys. Rev. D 90, 032001 (2014).

The MIPP number is N(pi+-)/POT *emerging from the target surface*, for a
NuMI **low-energy** target (47 graphite fins, 20 mm long x 6.4 mm wide x
18 mm tall, 2 interaction lengths), in bins of (pz, pT).

The g4numi side is taken from the `meson` tree of a zpNtuple, which records
*every* meson at the moment it is created, anywhere in the beamline.  Two
selections are therefore applied here:

  1. production vertex inside the target assembly (drops the horns, decay
     pipe, absorber and rock, which together dominate the inclusive count);
  2. an optional survival factor exp(-L/lambda_inel), where L is the graphite
     path length from the production point to the target surface along the
     pion direction.  This converts "produced in the target" into "emerges
     from the target", which is what MIPP measures.

Our run is the NOvA **medium-energy** target (48 fins, 24 mm long x 7.4 mm
wide x 63 mm tall), so the comparison is not like-for-like; see the report.

Usage:
    python compare_g4numi_mipp.py [--run r0003] [--lambda-cm 56.0]
"""
from __future__ import annotations

import os

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import uproot

RUNS = Path(os.environ.get("G4NUMI_RUNS_DIR", "runs"))  # g4numi runs: <RUNS>/rNNNN/zpNtuple_0NNN.root
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# NOvA medium-energy target geometry, from
# g4numi/src/NumiDataInput.cc (ME block, l.1035-1100) and
# g4numi/src/NumiNOvATarget.cc (fin placement, l.200-240).
# All lengths in cm, in the global g4numi frame (beam axis at x=y=0).
# ---------------------------------------------------------------------------
TARGET_Z0 = -143.3          # upstream face (/NuMI/det/set/targetPosition)
SEG_LEN = 2.40              # TargetSegLength
SEG_W = 0.74                # TargetSegWidth
SEG_H = 6.30                # TargetSegHeight
SEG_PITCH = 0.05            # TargetSegPitch
N_FINS = 48
BUDAL_HFVS_PITCH = 0.45
BUDAL_VFHS_PITCH = 0.50

# fin box in x,y: the beam sits SEG_W/2 below the top of the fin, so the
# graphite hangs down 6.3 cm and only 0.37 cm sits above the beam axis.
FIN_XLO, FIN_XHI = -SEG_W / 2, SEG_W / 2
FIN_YHI = -SEG_H / 2 + SEG_W / 2 + SEG_H / 2       # = +0.37
FIN_YLO = FIN_YHI - SEG_H                          # = -5.93

# Budal HFVS is the same fin rotated 90 deg about z, offset in x
HFVS_XLO = -SEG_H / 2 + SEG_W - SEG_H / 2
HFVS_XHI = -SEG_H / 2 + SEG_W + SEG_H / 2
HFVS_YLO, HFVS_YHI = -SEG_W / 2, SEG_W / 2

GRAPHITE_RHO = 1.78         # g/cm3, TargetDensity


def fin_boxes():
    """(xlo, xhi, ylo, yhi, zlo, zhi) for every graphite fin, in cm."""
    boxes = []
    z = TARGET_Z0
    boxes.append((HFVS_XLO, HFVS_XHI, HFVS_YLO, HFVS_YHI, z, z + SEG_LEN))
    z += SEG_LEN + BUDAL_HFVS_PITCH
    boxes.append((FIN_XLO, FIN_XHI, FIN_YLO, FIN_YHI, z, z + SEG_LEN))
    z += SEG_LEN + BUDAL_VFHS_PITCH
    for _ in range(N_FINS):
        boxes.append((FIN_XLO, FIN_XHI, FIN_YLO, FIN_YHI, z, z + SEG_LEN))
        z += SEG_LEN + SEG_PITCH
    return boxes


TARGET_Z1 = TARGET_Z0 + (N_FINS * SEG_LEN + (N_FINS - 1) * SEG_PITCH
                         + 2 * SEG_LEN + BUDAL_HFVS_PITCH + BUDAL_VFHS_PITCH)


# ---------------------------------------------------------------------------
# graphite path length from the production point to the target surface
# ---------------------------------------------------------------------------
def graphite_path_length(x, y, z, ux, uy, uz, boxes):
    """Path length (cm) through graphite along the unit direction (ux,uy,uz),
    starting at (x,y,z), summed over all fin boxes downstream of the start."""
    n = len(x)
    total = np.zeros(n)
    big = 1e9
    eps = 1e-12

    def slab(p, u, lo, hi):
        """Entry/exit path length for the slab lo<=p+u*t<=hi."""
        t0 = np.full(n, -big)
        t1 = np.full(n, big)
        moving = np.abs(u) > eps
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lo - p) / np.where(moving, u, 1.0)
            b = (hi - p) / np.where(moving, u, 1.0)
        lo_t = np.minimum(a, b)
        hi_t = np.maximum(a, b)
        t0 = np.where(moving, lo_t, np.where((p >= lo) & (p <= hi), -big, big))
        t1 = np.where(moving, hi_t, np.where((p >= lo) & (p <= hi), big, -big))
        return t0, t1

    for (xlo, xhi, ylo, yhi, zlo, zhi) in boxes:
        tx0, tx1 = slab(x, ux, xlo, xhi)
        ty0, ty1 = slab(y, uy, ylo, yhi)
        tz0, tz1 = slab(z, uz, zlo, zhi)
        t_in = np.maximum.reduce([tx0, ty0, tz0, np.zeros(n)])
        t_out = np.minimum.reduce([tx1, ty1, tz1])
        total += np.clip(t_out - t_in, 0.0, None)
    return total


# ---------------------------------------------------------------------------
def load_mipp(csv_path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v else None) for k, v in r.items()})
    return rows


def weighted_yield(w, mask, pot):
    """Yield per POT and its statistical error, for importance-weighted MC."""
    ww = w[mask]
    return ww.sum() / pot, np.sqrt((ww ** 2).sum()) / pot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r0003")
    ap.add_argument("--lambda-cm", type=float, default=56.0,
                    help="pion inelastic interaction length in graphite (cm)")
    ap.add_argument("--target-def", default="fins",
                    choices=["fins", "assembly"],
                    help="'fins' = graphite fin boxes only; "
                         "'assembly' = anything within r<5 cm of the beam axis")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    root = run_dir / f"zpNtuple_0{args.run[2:]}.root"
    f = uproot.open(root)
    pot = float(f["zpmeta"].arrays(library="np")["pots"][0])

    keys = sorted(k for k in f.keys() if k.startswith("meson;"))
    tname = keys[-1]                       # highest TTree cycle
    a = f[tname].arrays(["pdg", "px", "py", "pz", "E", "x", "y", "z",
                         "weight", "tgen"], library="np")

    pdg = a["pdg"]
    is_pi = np.abs(pdg) == 211
    inz = (a["z"] >= TARGET_Z0) & (a["z"] <= TARGET_Z1)
    sel = is_pi & inz

    # neutral mesons produced in the same target volume (for the pi0
    # normalisation that MIPP constrains only indirectly)
    boxes_all = fin_boxes()
    in_fins = np.zeros(len(pdg), dtype=bool)
    for (xlo, xhi, ylo, yhi, zlo, zhi) in boxes_all:
        in_fins |= ((a["x"] >= xlo) & (a["x"] <= xhi)
                    & (a["y"] >= ylo) & (a["y"] <= yhi)
                    & (a["z"] >= zlo) & (a["z"] <= zhi))
    neutrals = {}
    for code, name in ((111, "pi0"), (221, "eta"), (211, "pip"), (-211, "pim")):
        m = in_fins & (pdg == code)
        neutrals[name] = {
            "in_target": float(a["weight"][m].sum() / pot),
            "whole_beamline": float(a["weight"][pdg == code].sum() / pot),
        }

    x, y, z = a["x"][sel], a["y"][sel], a["z"][sel]
    px, py, pz = a["px"][sel], a["py"][sel], a["pz"][sel]
    w, q = a["weight"][sel], np.sign(pdg[sel])

    boxes = fin_boxes()
    if args.target_def == "fins":
        in_tgt = np.zeros(len(x), dtype=bool)
        for (xlo, xhi, ylo, yhi, zlo, zhi) in boxes:
            in_tgt |= ((x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
                       & (z >= zlo) & (z <= zhi))
    else:
        in_tgt = np.hypot(x, y) < 5.0

    x, y, z, px, py, pz, w, q = (v[in_tgt] for v in (x, y, z, px, py, pz, w, q))

    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    pt = np.hypot(px, py)
    good = p > 0
    ux, uy, uz = px / p, py / p, pz / p

    L = graphite_path_length(x, y, z, ux, uy, uz, boxes)
    surv = np.exp(-L / args.lambda_cm)

    pos, neg = q > 0, q < 0
    summary = {
        "run": args.run,
        "pot": pot,
        "root": str(root),
        "target_def": args.target_def,
        "lambda_cm": args.lambda_cm,
        "target_z_cm": [TARGET_Z0, TARGET_Z1],
        "inclusive_all_beamline": {
            "pip": float(a["weight"][pdg == 211].sum() / pot),
            "pim": float(a["weight"][pdg == -211].sum() / pot),
        },
        "produced_in_target": {
            "pip": float(w[pos].sum() / pot),
            "pim": float(w[neg].sum() / pot),
        },
        "emerging_from_target": {
            "pip": float((w * surv)[pos].sum() / pot),
            "pim": float((w * surv)[neg].sum() / pot),
        },
        "mean_graphite_path_cm": float(np.average(L, weights=w)),
        "meson_yields": neutrals,
        "pi0_over_mean_charged_in_target": float(
            neutrals["pi0"]["in_target"]
            / (0.5 * (neutrals["pip"]["in_target"] + neutrals["pim"]["in_target"]))),
        "target_fraction_of_beamline_pi0": float(
            neutrals["pi0"]["in_target"] / neutrals["pi0"]["whole_beamline"]),
    }

    # ---- bin-by-bin against MIPP ------------------------------------------
    mipp = load_mipp(Path(os.environ.get("MIPP_TABLE", str(HERE / "mipp_table1.csv"))))
    rows = []
    for m in mipp:
        b = ((pz >= m["pz_lo"]) & (pz < m["pz_hi"])
             & (pt >= m["pt_lo"]) & (pt < m["pt_hi"]))
        row = dict(m)
        for lab, weights in (("prod", w), ("emerg", w * surv)):
            for chg, cm in (("pip", pos), ("pim", neg)):
                val, err = weighted_yield(weights, b & cm, pot)
                row[f"g4_{lab}_{chg}"] = val
                row[f"g4_{lab}_{chg}_stat"] = err
            num, den = row[f"g4_{lab}_pim"], row[f"g4_{lab}_pip"]
            row[f"g4_{lab}_ratio"] = num / den if den > 0 else None
        rows.append(row)

    # totals over the MIPP-measured phase space only
    for chg, cm, key in (("pip", pos, "n_pip"), ("pim", neg, "n_pim")):
        acc = np.zeros(len(pz), dtype=bool)
        mipp_sum = 0.0
        mipp_err2 = 0.0
        for m in mipp:
            if m[key] is None:
                continue
            acc |= ((pz >= m["pz_lo"]) & (pz < m["pz_hi"])
                    & (pt >= m["pt_lo"]) & (pt < m["pt_hi"]))
            mipp_sum += m[key]
            stat = m[key] * m[f"d_{chg}_stat"] / 100.0
            syst = m[key] * m[f"d_{chg}_syst"] / 100.0
            mipp_err2 += stat ** 2 + syst ** 2
        pv, pe = weighted_yield(w, acc & cm, pot)
        ev, ee = weighted_yield(w * surv, acc & cm, pot)
        summary.setdefault("mipp_acceptance", {})[chg] = {
            "mipp": mipp_sum,
            "mipp_err_uncorrelated": float(np.sqrt(mipp_err2)),
            "g4_produced": pv, "g4_produced_stat": pe,
            "g4_emerging": ev, "g4_emerging_stat": ee,
            "ratio_produced": pv / mipp_sum,
            "ratio_emerging": ev / mipp_sum,
        }

    prefix = args.out_prefix or f"{args.run}_{args.target_def}"
    out_csv = HERE / f"compare_{prefix}.csv"
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    out_json = HERE / f"summary_{prefix}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    main()
