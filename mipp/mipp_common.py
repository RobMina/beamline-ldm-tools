#!/usr/bin/env python
"""Shared pieces for the g4numi <-> MIPP comparison plots.

Everything is expressed on MIPP's own (pz, pT) grid (PRD 90, 032001, Table I):
24 pz bins from 0.30 to 80 GeV/c -- note the 2-4 GeV/c bin is never measured --
and 6 pT bins from 0 to 2 GeV/c.

"MIPP prediction" below means the pi0 flux the reference analysis produces
(NOvA_pion_dist.ipynb): the analytic BMPT pi0 source reweighted bin-by-bin to
MIPP's charged-pion average, so that by construction it *equals*
(N(pi+) + N(pi-))/2 inside MIPP's bins.
"""
from __future__ import annotations

import os

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("G4NUMI_RUNS_DIR", "runs"))  # g4numi runs: <RUNS>/rNNNN/zpNtuple_0NNN.root
DATA = Path(os.environ.get("G4NUMI_LDM_DATA", str(HERE / "data")))  # slim .npz extracts

# g4numi ME target extent in the global frame (NumiDataInput.cc ME block)
TARGET_Z0, TARGET_Z1 = -143.3, -20.0

# Beamline landmarks, cm. Values from the g4numi geometry / zpNtuple notes.
BEAMLINE_REGIONS = [
    ("upstream / baffle", -1e9, TARGET_Z0),
    ("target", TARGET_Z0, TARGET_Z1),
    ("horn 1", TARGET_Z1, 400.0),
    ("horn 2 + target hall", 400.0, 4500.0),
    ("decay pipe", 4500.0, 67500.0),
    ("decay-pipe end / shielding", 67500.0, 72000.0),
    ("hadron absorber", 72000.0, 72800.0),
    ("rock / muon alcoves", 72800.0, 1e9),
]


def mipp_rows():
    with open(Path(os.environ.get("MIPP_TABLE", str(HERE / "mipp_table1.csv")))) as fh:
        return [{k: (float(v) if v else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def mipp_bins():
    rows = mipp_rows()
    pz = np.array(sorted({r["pz_lo"] for r in rows} | {r["pz_hi"] for r in rows}))
    pt = np.array(sorted({r["pt_lo"] for r in rows} | {r["pt_hi"] for r in rows}))
    return pz, pt


def mipp_grids():
    """MIPP yields on the (pz, pT) grid.  Unmeasured cells are NaN.

    Returns a dict with 'pip', 'pim', 'avg' (the pi0 proxy the reference analysis
    builds) and matching fractional uncertainties 'pip_err' etc.
    """
    bins_pz, bins_pt = mipp_bins()
    shape = (len(bins_pz) - 1, len(bins_pt) - 1)
    out = {k: np.full(shape, np.nan) for k in
           ("pip", "pim", "avg", "pip_err", "pim_err", "avg_err")}
    for r in mipp_rows():
        i = int(np.searchsorted(bins_pz, r["pz_lo"], side="right") - 1)
        j = int(np.searchsorted(bins_pt, r["pt_lo"], side="right") - 1)
        for chg in ("pip", "pim"):
            v = r[f"n_{chg}"]
            if v is None:
                continue
            out[chg][i][j] = v
            out[f"{chg}_err"][i][j] = v * np.hypot(
                r[f"d_{chg}_stat"], r[f"d_{chg}_syst"]) / 100.0
        # notebook convention: average pi+ and pi-, and take the larger syst
        if r["n_pip"] is not None and r["n_pim"] is not None:
            out["avg"][i][j] = 0.5 * (r["n_pip"] + r["n_pim"])
            syst = max(r["d_pip_syst"], r["d_pim_syst"]) / 100.0
            stat = 0.5 * np.hypot(r["n_pip"] * r["d_pip_stat"] / 100.0,
                                  r["n_pim"] * r["d_pim_stat"] / 100.0)
            out["avg_err"][i][j] = np.hypot(stat, out["avg"][i][j] * syst)
    return out


def grid_from(pz, pt, w, pot):
    """Histogram a weighted particle sample onto the MIPP grid, per POT."""
    bins_pz, bins_pt = mipp_bins()
    h, _, _ = np.histogram2d(pz, pt, bins=[bins_pz, bins_pt], weights=w)
    return h / pot


def load_run(source):
    """Load a g4numi meson sample.

    `source` is either a run id under $G4NUMI_RUNS_DIR (read
    straight from the zpNtuple; production tree only) or the stem of a slimmed
    .npz in data/ (carries both the production and target-exit trees).
    """
    if (DATA / f"{source}.npz").exists():
        d = np.load(DATA / f"{source}.npz")
        out = {"pot": float(d["pot"]), "trees": {}}
        for tag in ("prod", "exit"):
            if f"{tag}_pdg" not in d:
                continue
            out["trees"][tag] = {k: d[f"{tag}_{k}"].astype(
                np.int32 if k in ("pdg", "tgen") else np.float64)
                for k in ("pdg", "px", "py", "pz", "x", "y", "z", "weight", "tgen")}
        return out

    import uproot
    root = (RUNS / source
            / f"zpNtuple_0{source[2:]}.root")
    f = uproot.open(root)
    pot = float(f["zpmeta"].arrays(library="np")["pots"][0])
    tname = sorted(k for k in f.keys() if k.startswith("meson;"))[-1]
    a = f[tname].arrays(["pdg", "px", "py", "pz", "x", "y", "z", "weight", "tgen"],
                        library="np")
    keep = np.isin(a["pdg"], (211, -211, 111, 221))
    return {"pot": pot,
            "trees": {"prod": {k: v[keep] for k, v in a.items()}}}


def in_target(tree, model="me"):
    """Production vertex inside the graphite of the target.

    The transverse footprint differs between the two targets: the ME fin hangs
    6.3 cm below the beam with only 0.37 cm above it, the LE fin is 1.8 cm tall
    and centred.  z extent is taken from the fin train.
    """
    geom = {"me": dict(w=0.74, ylo=-5.93, yhi=0.37, z=(-143.3, -20.0)),
            "le": dict(w=0.64, ylo=-0.90, yhi=0.90, z=(-143.3, -45.4))}[model]
    x, y, z = tree["x"], tree["y"], tree["z"]
    return ((np.abs(x) <= geom["w"] / 2) & (y >= geom["ylo"]) & (y <= geom["yhi"])
            & (z >= geom["z"][0]) & (z <= geom["z"][1]))


def select(tree, pdg=None, mask=None):
    m = np.ones(len(tree["pdg"]), dtype=bool) if mask is None else mask.copy()
    if pdg is not None:
        m &= tree["pdg"] == pdg
    pt = np.hypot(tree["px"][m], tree["py"][m])
    return tree["pz"][m], pt, tree["weight"][m]


CONFIGS = {
    "me_nom": dict(label="ME target, nominal NOvA beam ($\\sigma$=1.4 mm)",
                   short="ME / nominal", model="me"),
    "me_exit": dict(label="ME target, MIPP beam ($\\sigma$=0.11 mm)",
                    short="ME / MIPP beam", model="me"),
    "le_exit": dict(label="LE target, MIPP beam ($\\sigma$=0.11 mm)"
                          " — MIPP-comparison configuration",
                    short="LE / MIPP beam", model="le"),
    "r0003": dict(label="ME target, nominal NOvA beam ($\\sigma$=1.4 mm), $10^6$ POT",
                  short="ME / nominal", model="me"),
}
