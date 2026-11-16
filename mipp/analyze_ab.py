#!/usr/bin/env python
"""LE-vs-ME NuMI target A/B for the MIPP comparison.

Reads the slimmed charged-pion arrays produced on the cluster by
`slim.py` (see the report) for two g4numi runs that differ only in the
target geometry:

    me_pencil  NOvA medium-energy target (48 fins, 24 x 7.4 x 63 mm)
    le_pencil  NuMI low-energy target    (47 fins, 20 x 6.4 x 18 mm)  <- MIPP

Both use the MIPP beam spot (sigma = 0.11 mm).  The fin z origin is fitted
from the production vertices themselves, so the model is validated rather
than assumed.

Usage:
    python analyze_ab.py me_pencil_slim.npz le_pencil_slim.npz
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
from compare_g4numi_mipp import graphite_path_length  # noqa: E402

HERE = Path(__file__).resolve().parent
LAMBDA_CM = 56.0

# (n_fins, seg_len, seg_width, seg_height, pitch, y_top) in cm.
# y_top is the top of the fin relative to the beam axis: the ME fin hangs
# 6.3 cm below the beam with only 0.37 cm above it; the LE fin is centred.
GEOM = {
    "me": dict(n=48, ln=2.40, w=0.74, h=6.30, pitch=0.05, y_top=0.37),
    "le": dict(n=47, ln=2.00, w=0.64, h=1.80, pitch=0.03, y_top=0.90),
}


def find_extent(model, x, y, z, w, bin_cm=0.5, thresh=0.01):
    """Locate the fin train in z from the production vertices themselves:
    take the longest contiguous run of z bins holding more than `thresh` of the
    peak density inside the fin's transverse footprint.  This picks up the
    graphite column (and, for the ME target, its two abutting Budal monitor
    fins) while rejecting the LE Budal monitor, which sits ~15 cm upstream
    behind a large gap."""
    g = GEOM[model]
    core = (np.abs(x) < g["w"] / 2) & (y > g["y_top"] - g["h"]) & (y < g["y_top"])
    zc, wc = z[core], w[core]
    edges = np.arange(zc.min() - bin_cm, zc.max() + 2 * bin_cm, bin_cm)
    hist, _ = np.histogram(zc, bins=edges, weights=wc)
    full = hist > thresh * hist.max()
    best = (0, 0, 0)
    i = 0
    while i < len(full):
        if full[i]:
            j = i
            while j + 1 < len(full) and full[j + 1]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        i += 1
    _, i, j = best
    return float(edges[i]), float(edges[j + 1])


def target_box(model, z_lo, z_hi):
    g = GEOM[model]
    return [(-g["w"] / 2, g["w"] / 2, g["y_top"] - g["h"], g["y_top"], z_lo, z_hi)]


def duty(model):
    """Graphite fraction along z (the inter-fin pitch gaps carry no material)."""
    g = GEOM[model]
    return g["n"] * g["ln"] / (g["n"] * g["ln"] + (g["n"] - 1) * g["pitch"])


def load_mipp():
    with open(Path(os.environ.get("MIPP_TABLE", str(HERE / "mipp_table1.csv")))) as fh:
        return [{k: (float(v) if v else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def analyse(npz_path, model):
    d = np.load(npz_path)
    pot = float(d["pot"])
    x, y, z = d["x"].astype(float), d["y"].astype(float), d["z"].astype(float)
    px, py, pz = (d[k].astype(float) for k in ("px", "py", "pz"))
    w, pdg = d["weight"].astype(float), d["pdg"]

    z_lo, z_hi = find_extent(model, x, y, z, w)
    boxes = target_box(model, z_lo, z_hi)
    (xlo, xhi, ylo, yhi, zlo, zhi) = boxes[0]
    in_tgt = ((x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
              & (z >= zlo) & (z <= zhi))

    x, y, z, px, py, pz, w, pdg = (v[in_tgt] for v in
                                   (x, y, z, px, py, pz, w, pdg))
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    pt = np.hypot(px, py)
    L = graphite_path_length(x, y, z, px / p, py / p, pz / p, boxes) * duty(model)
    surv = np.exp(-L / LAMBDA_CM)
    pos, neg = pdg > 0, pdg < 0

    res = dict(npz=str(npz_path), model=model, pot=pot,
               target_z_cm=[z_lo, z_hi],
               graphite_cm=GEOM[model]["n"] * GEOM[model]["ln"],
               produced={"pip": w[pos].sum() / pot, "pim": w[neg].sum() / pot},
               emerging={"pip": (w * surv)[pos].sum() / pot,
                         "pim": (w * surv)[neg].sum() / pot})

    mipp = load_mipp()
    per_bin = {}
    for chg, cm, key in (("pip", pos, "n_pip"), ("pim", neg, "n_pim")):
        acc = np.zeros(len(pz), dtype=bool)
        tot = 0.0
        for m in mipp:
            if m[key] is None:
                continue
            acc |= ((pz >= m["pz_lo"]) & (pz < m["pz_hi"])
                    & (pt >= m["pt_lo"]) & (pt < m["pt_hi"]))
            tot += m[key]
        res.setdefault("mipp_acceptance", {})[chg] = {
            "mipp": tot,
            "produced": w[acc & cm].sum() / pot,
            "produced_stat": np.sqrt((w[acc & cm] ** 2).sum()) / pot,
            "emerging": (w * surv)[acc & cm].sum() / pot,
            "emerging_stat": np.sqrt(((w * surv)[acc & cm] ** 2).sum()) / pot,
        }
    agg = defaultdict(lambda: np.zeros(4))
    for m in mipp:
        b = ((pz >= m["pz_lo"]) & (pz < m["pz_hi"])
             & (pt >= m["pt_lo"]) & (pt < m["pt_hi"]))
        a = agg[(m["pz_lo"], m["pz_hi"])]
        a += [w[b & pos].sum() / pot, w[b & neg].sum() / pot,
              (w * surv)[b & pos].sum() / pot, (w * surv)[b & neg].sum() / pot]
    per_bin = {f"{k[0]}-{k[1]}": list(v) for k, v in sorted(agg.items())}
    res["per_pz_bin"] = per_bin
    return res


def main():
    me = analyse(sys.argv[1], "me")
    le = analyse(sys.argv[2], "le")
    out = {"me": me, "le": le}

    print(f"{'':22s} {'ME (NOvA)':>12s} {'LE (MIPP)':>12s} {'ME/LE':>8s}")
    print(f"{'graphite length (cm)':22s} {me['graphite_cm']:12.1f} "
          f"{le['graphite_cm']:12.1f} {me['graphite_cm']/le['graphite_cm']:8.3f}")
    print(f"{'fitted z extent (cm)':22s} "
          f"{str([round(v, 1) for v in me['target_z_cm']]):>12s} "
          f"{str([round(v, 1) for v in le['target_z_cm']]):>12s}")
    for stage in ("produced", "emerging"):
        for chg in ("pip", "pim"):
            a, b = me[stage][chg], le[stage][chg]
            print(f"{stage+' '+chg+' /POT':22s} {a:12.4f} {b:12.4f} {a/b:8.3f}")
    print("\n-- inside the MIPP-measured (pz,pT) bins --")
    for chg in ("pip", "pim"):
        m, l = me["mipp_acceptance"][chg], le["mipp_acceptance"][chg]
        print(f"  {chg}: MIPP data {m['mipp']:.4f}")
        for stage in ("produced", "emerging"):
            print(f"     {stage:9s} ME {m[stage]:.4f} (data/MC-ratio "
                  f"{m[stage]/m['mipp']:.3f})   LE {l[stage]:.4f} "
                  f"({l[stage]/l['mipp']:.3f})   ME/LE {m[stage]/l[stage]:.3f}")

    (HERE / "ab_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", HERE / "ab_summary.json")


if __name__ == "__main__":
    main()
