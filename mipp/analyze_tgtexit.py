#!/usr/bin/env python
"""Compare g4numi's truth-level target-exit flux with MIPP, and with the
analytic exp(-L/lambda) escape model it replaces.

Input is the slimmed output of `slim2.py`, which carries two sets of charged
pions from the same run:

    prod_*   the `meson` tree      - every pion where it was created
    exit_*   the `tgtexit` tree    - every pion crossing the outer surface of
                                     the target assembly (added to
                                     NumiSteppingAction; enable with
                                     /NuMI/output/outputTgtExitTree true)

The exit flux is what MIPP measures, so it needs no absorption modelling at all.

Usage:
    python analyze_tgtexit.py me_exit_slim.npz le_exit_slim.npz
"""
from __future__ import annotations

import os

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ab import GEOM, LAMBDA_CM, duty, find_extent, target_box  # noqa: E402
from compare_g4numi_mipp import graphite_path_length  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_mipp():
    with open(Path(os.environ.get("MIPP_TABLE", str(HERE / "mipp_table1.csv")))) as fh:
        return [{k: (float(v) if v else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def yields(pz, pt, w, pos, neg, mipp):
    """Per-charge totals inside the MIPP-measured bins, and per-pz-bin sums."""
    out = {}
    for chg, cm, key in (("pip", pos, "n_pip"), ("pim", neg, "n_pim")):
        acc = np.zeros(len(pz), dtype=bool)
        tot = 0.0
        for m in mipp:
            if m[key] is None:
                continue
            acc |= ((pz >= m["pz_lo"]) & (pz < m["pz_hi"])
                    & (pt >= m["pt_lo"]) & (pt < m["pt_hi"]))
            tot += m[key]
        out[chg] = {"mipp": tot, "g4": float(w[acc & cm].sum()),
                    "stat": float(np.sqrt((w[acc & cm] ** 2).sum()))}
    return out


def analyse(npz_path, model):
    d = np.load(npz_path)
    pot = float(d["pot"])
    mipp = load_mipp()
    res = {"npz": str(npz_path), "model": model, "pot": pot}

    # ---- production side, plus the analytic escape model for reference -----
    x, y, z = (d[f"prod_{k}"].astype(float) for k in ("x", "y", "z"))
    px, py, pz = (d[f"prod_{k}"].astype(float) for k in ("px", "py", "pz"))
    w, pdg = d["prod_weight"].astype(float), d["prod_pdg"]
    z_lo, z_hi = find_extent(model, x, y, z, w)
    boxes = target_box(model, z_lo, z_hi)
    (xlo, xhi, ylo, yhi, zl, zh) = boxes[0]
    m = ((x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
         & (z >= zl) & (z <= zh))
    x, y, z, px, py, pz, w, pdg = (v[m] for v in (x, y, z, px, py, pz, w, pdg))
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    pt = np.hypot(px, py)
    L = graphite_path_length(x, y, z, px / p, py / p, pz / p, boxes) * duty(model)
    surv = np.exp(-L / LAMBDA_CM)
    pos, neg = pdg > 0, pdg < 0

    res["target_z_cm"] = [z_lo, z_hi]
    res["produced"] = {"pip": float(w[pos].sum() / pot),
                       "pim": float(w[neg].sum() / pot)}
    res["model_emerging"] = {"pip": float((w * surv)[pos].sum() / pot),
                             "pim": float((w * surv)[neg].sum() / pot)}
    res["mipp_bins_produced"] = yields(pz, pt, w / pot, pos, neg, mipp)
    res["mipp_bins_model_emerging"] = yields(pz, pt, w * surv / pot, pos, neg, mipp)
    prod_bins = defaultdict(lambda: np.zeros(2))
    for mb in mipp:
        b = ((pz >= mb["pz_lo"]) & (pz < mb["pz_hi"])
             & (pt >= mb["pt_lo"]) & (pt < mb["pt_hi"]))
        prod_bins[(mb["pz_lo"], mb["pz_hi"])] += [w[b & pos].sum() / pot,
                                                  w[b & neg].sum() / pot]

    # ---- truth target-exit flux -------------------------------------------
    epx, epy, epz = (d[f"exit_{k}"].astype(float) for k in ("px", "py", "pz"))
    ew, epdg = d["exit_weight"].astype(float), d["exit_pdg"]
    ept = np.hypot(epx, epy)
    epos, eneg = epdg > 0, epdg < 0
    res["exiting"] = {"pip": float(ew[epos].sum() / pot),
                      "pim": float(ew[eneg].sum() / pot)}
    res["mipp_bins_exiting"] = yields(epz, ept, ew / pot, epos, eneg, mipp)
    exit_bins = defaultdict(lambda: np.zeros(2))
    for mb in mipp:
        b = ((epz >= mb["pz_lo"]) & (epz < mb["pz_hi"])
             & (ept >= mb["pt_lo"]) & (ept < mb["pt_hi"]))
        exit_bins[(mb["pz_lo"], mb["pz_hi"])] += [ew[b & epos].sum() / pot,
                                                  ew[b & eneg].sum() / pot]
    res["per_pz_bin"] = {f"{k[0]}-{k[1]}": {"produced": list(prod_bins[k]),
                                            "exiting": list(exit_bins[k])}
                         for k in sorted(prod_bins)}
    return res


def main():
    me = analyse(sys.argv[1], "me")
    le = analyse(sys.argv[2], "le")

    print(f"{'per POT':28s} {'ME (NOvA)':>11s} {'LE (MIPP)':>11s} {'ME/LE':>7s}")
    for stage in ("produced", "model_emerging", "exiting"):
        for chg in ("pip", "pim"):
            a, b = me[stage][chg], le[stage][chg]
            print(f"  {stage+' '+chg:26s} {a:11.4f} {b:11.4f} {a/b:7.3f}")
    print("\n  escape fraction (exiting/produced)")
    for tag, r in (("ME", me), ("LE", le)):
        for chg in ("pip", "pim"):
            truth = r["exiting"][chg] / r["produced"][chg]
            modl = r["model_emerging"][chg] / r["produced"][chg]
            print(f"    {tag} {chg}: truth {truth:.3f}   exp(-L/lambda) model "
                  f"{modl:.3f}   model/truth {modl/truth:.3f}")

    print("\n-- inside the MIPP-measured (pz,pT) bins --")
    for tag, r in (("ME", me), ("LE (like-for-like vs MIPP)", le)):
        print(f"  {tag}:")
        for chg in ("pip", "pim"):
            mb = r["mipp_bins_exiting"][chg]
            pb = r["mipp_bins_produced"][chg]
            eb = r["mipp_bins_model_emerging"][chg]
            print(f"    {chg}: MIPP {mb['mipp']:.4f} | produced {pb['g4']:.4f} "
                  f"({pb['g4']/pb['mipp']:.3f}) | model-emerging {eb['g4']:.4f} "
                  f"({eb['g4']/eb['mipp']:.3f}) | TRUTH-exiting {mb['g4']:.4f} "
                  f"({mb['g4']/mb['mipp']:.3f})")

    print("\n-- LE target, truth exit flux vs MIPP, by pz --")
    mipp = load_mipp()
    agg = defaultdict(lambda: np.zeros(2))
    for m in mipp:
        a = agg[(m["pz_lo"], m["pz_hi"])]
        a += [m["n_pip"] or 0.0, m["n_pim"] or 0.0]
    print(f"{'pz bin':>14} {'MIPP pi+':>9} {'g4 exit':>9} {'ratio':>7} "
          f"| {'MIPP pi-':>9} {'g4 exit':>9} {'ratio':>7}")
    for k in sorted(agg):
        mp, mm = agg[k]
        gp, gm = le["per_pz_bin"][f"{k[0]}-{k[1]}"]["exiting"]
        print(f"[{k[0]:5.1f},{k[1]:5.1f}) {mp:9.4f} {gp:9.4f} {gp/mp:7.3f} "
              f"| {mm:9.4f} {gm:9.4f} {gm/mm if mm else float('nan'):7.3f}")

    (HERE / "tgtexit_summary.json").write_text(
        json.dumps({"me": me, "le": le}, indent=2, default=float))
    print("\nwrote", HERE / "tgtexit_summary.json")


if __name__ == "__main__":
    main()
