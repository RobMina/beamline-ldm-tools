#!/usr/bin/env python
"""g4numi vs the MIPP prediction on MIPP's own (pz, pT) grid.

Three comparisons, each produced for every available g4numi configuration:

  1  pi0  -- all g4numi pi0, produced anywhere in the beamline
  2  pi0  -- only g4numi pi0 produced in the target
  3  pi+/pi- -- g4numi pions that *exit* the target (truth `tgtexit` tree)

"MIPP prediction" is the pi0 flux the reference analysis builds by reweighting the
analytic BMPT source to MIPP's charged-pion average, so inside MIPP's bins it
equals (N(pi+) + N(pi-))/2.  For comparison 3 the reference is the directly
measured N(pi+) or N(pi-).

Bins are drawn in index space, not to scale: MIPP's binning spans 0.3-80 GeV/c
in pz and its last pT bin is 4x wider than the rest, so a to-scale axis would
render most of the grid invisible.

Usage:
    python plot_mipp_grid.py                 # every config it finds
    python plot_mipp_grid.py --configs le_exit
"""
from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

import mipp_common as mc

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("G4NUMI_LDM_OUT", str(HERE / "plots")))


def _tick_labels(edges, every=1):
    return [f"{e:g}" for e in edges[::every]]


def _draw_map(ax, grid, bins_pz, bins_pt, title, *, ratio=False, cbar_label=""):
    masked = np.ma.masked_invalid(grid)
    if ratio:
        hi = float(np.nanpercentile(grid, 98)) if np.isfinite(grid).any() else 2.5
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(hi, 1.6))
        im = ax.pcolormesh(np.arange(len(bins_pz)), np.arange(len(bins_pt)),
                           masked.T, cmap="RdBu_r", norm=norm,
                           edgecolors="0.85", linewidth=0.3)
    else:
        pos = masked.compressed()
        pos = pos[pos > 0]
        norm = LogNorm(vmin=max(pos.min(), 1e-6), vmax=pos.max()) if len(pos) else None
        im = ax.pcolormesh(np.arange(len(bins_pz)), np.arange(len(bins_pt)),
                           masked.T, cmap="viridis", norm=norm,
                           edgecolors="0.85", linewidth=0.3)
    ax.set_xticks(np.arange(len(bins_pz))[::3])
    ax.set_xticklabels(_tick_labels(bins_pz, 3), fontsize=7, rotation=45)
    ax.set_yticks(np.arange(len(bins_pt)))
    ax.set_yticklabels(_tick_labels(bins_pt), fontsize=7)
    ax.set_xlabel(r"$p_z$ (GeV/c)", fontsize=8)
    ax.set_ylabel(r"$p_T$ (GeV/c)", fontsize=8)
    ax.set_title(title, fontsize=9)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    return im


def _projection(ax_main, ax_ratio, edges, g4, ref, ref_err, xlabel, logx,
                ref_name="MIPP prediction"):
    """1D projection with a ratio panel underneath.  Bins with no MIPP
    measurement are dropped from both, so the ratio is like-for-like."""
    ok = np.isfinite(ref) & (ref > 0)
    cen = 0.5 * (edges[:-1] + edges[1:])
    wid = 0.5 * (edges[1:] - edges[:-1])
    ax_main.errorbar(cen[ok], ref[ok], yerr=ref_err[ok], xerr=wid[ok], fmt="o",
                     ms=3.5, lw=1.1, color="k", label=ref_name)
    ax_main.errorbar(cen[ok], g4[ok], xerr=wid[ok], fmt="s", ms=3.5, lw=1.1,
                     color="tab:red", label="g4numi")
    ax_main.set_yscale("log")
    if logx:
        ax_main.set_xscale("log")
        ax_ratio.set_xscale("log")
    ax_main.set_ylabel("yield / POT per bin", fontsize=8)
    ax_main.grid(alpha=0.25, which="both")
    ax_main.legend(fontsize=7)
    ax_main.tick_params(labelbottom=False, labelsize=7)

    r = g4[ok] / ref[ok]
    rerr = ref_err[ok] / ref[ok]
    ax_ratio.fill_between(cen[ok], 1 - rerr, 1 + rerr, step="mid",
                          color="0.75", alpha=0.6, label="MIPP unc.")
    ax_ratio.errorbar(cen[ok], r, xerr=wid[ok], fmt="s", ms=3.5, lw=1.1,
                      color="tab:red")
    ax_ratio.axhline(1.0, color="k", lw=0.8)
    ax_ratio.set_ylim(0, max(2.0, 1.18 * float(np.nanmax(r))))
    ax_ratio.set_xlabel(xlabel, fontsize=8)
    ax_ratio.set_ylabel(f"g4numi / {ref_name.replace('MIPP prediction','MIPP')}",
                        fontsize=8)
    ax_ratio.grid(alpha=0.25, which="both")
    ax_ratio.tick_params(labelsize=7)


def make_figure(g4grid, refgrid, referr, title, subtitle, outfile,
                ref_name="MIPP prediction"):
    bins_pz, bins_pt = mc.mipp_bins()
    # compare only where MIPP has a measurement
    measured = np.isfinite(refgrid) & (refgrid > 0)
    g4m = np.where(measured, g4grid, np.nan)
    refm = np.where(measured, refgrid, np.nan)
    ratio = np.where(measured, g4grid / np.where(measured, refgrid, 1), np.nan)

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1.0, 0.55],
                          hspace=0.42, wspace=0.55,
                          left=0.06, right=0.97, top=0.87, bottom=0.09)
    _draw_map(fig.add_subplot(gs[0, 0:2]), g4m, bins_pz, bins_pt,
              "g4numi", cbar_label="yield / POT per bin")
    _draw_map(fig.add_subplot(gs[0, 2:4]), refm, bins_pz, bins_pt,
              ref_name, cbar_label="yield / POT per bin")
    _draw_map(fig.add_subplot(gs[0, 4:6]), ratio, bins_pz, bins_pt,
              f"ratio  g4numi / {ref_name}", ratio=True, cbar_label="ratio")

    # projections, summing only over measured cells
    g4_pz = np.nansum(np.where(measured, g4grid, np.nan), axis=1)
    ref_pz = np.nansum(np.where(measured, refgrid, np.nan), axis=1)
    err_pz = np.sqrt(np.nansum(np.where(measured, referr, np.nan) ** 2, axis=1))
    g4_pt = np.nansum(np.where(measured, g4grid, np.nan), axis=0)
    ref_pt = np.nansum(np.where(measured, refgrid, np.nan), axis=0)
    err_pt = np.sqrt(np.nansum(np.where(measured, referr, np.nan) ** 2, axis=0))

    _projection(fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[2, 0:3]),
                bins_pz, g4_pz, ref_pz, err_pz, r"$p_z$ (GeV/c)", True, ref_name)
    _projection(fig.add_subplot(gs[1, 3:6]), fig.add_subplot(gs[2, 3:6]),
                bins_pt, g4_pt, ref_pt, err_pt, r"$p_T$ (GeV/c)", False, ref_name)

    tot_g4 = np.nansum(g4m)
    tot_ref = np.nansum(refm)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    fig.text(0.5, 0.905,
             f"summed over MIPP's measured bins:  g4numi {tot_g4:.3f}/POT   "
             f"{ref_name} {tot_ref:.3f}/POT   ratio {tot_g4/tot_ref:.3f}",
             ha="center", fontsize=9, color="tab:red")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / outfile, dpi=140)
    plt.close(fig)
    print("wrote", OUT / outfile)
    return {"g4": float(tot_g4), "ref": float(tot_ref),
            "ratio": float(tot_g4 / tot_ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()

    available = [c for c in ("r0003", "me_nom", "me_exit", "le_exit")
                 if (mc.DATA / f"{c}.npz").exists()
                 or (mc.RUNS / c).exists()]
    configs = args.configs or available
    mipp = mc.mipp_grids()
    summary = {}

    for cfg in configs:
        meta = mc.CONFIGS[cfg]
        run = mc.load_run(cfg)
        pot, trees = run["pot"], run["trees"]
        summary[cfg] = {}

        # ---- (1) all pi0, anywhere in the beamline -------------------------
        pz, pt, w = mc.select(trees["prod"], pdg=111)
        summary[cfg]["pi0_all"] = make_figure(
            mc.grid_from(pz, pt, w, pot), mipp["avg"], mipp["avg_err"],
            r"(1)  $\pi^0$ produced anywhere in the beamline  vs  MIPP prediction",
            meta["label"], f"fig1_pi0_beamline_{cfg}.png")

        # ---- (2) pi0 produced in the target --------------------------------
        m = mc.in_target(trees["prod"], meta["model"]) & (trees["prod"]["pdg"] == 111)
        pz, pt, w = mc.select(trees["prod"], mask=m)
        summary[cfg]["pi0_target"] = make_figure(
            mc.grid_from(pz, pt, w, pot), mipp["avg"], mipp["avg_err"],
            r"(2)  $\pi^0$ produced in the target only  vs  MIPP prediction",
            meta["label"], f"fig2_pi0_target_{cfg}.png")

        # ---- (3) charged pions leaving the target --------------------------
        if "exit" not in trees:
            print(f"  [{cfg}] no tgtexit tree, skipping comparison 3")
            continue
        for code, name, sym in ((211, "pip", r"\pi^+"), (-211, "pim", r"\pi^-")):
            pz, pt, w = mc.select(trees["exit"], pdg=code)
            summary[cfg][f"{name}_exit"] = make_figure(
                mc.grid_from(pz, pt, w, pot), mipp[name], mipp[f"{name}_err"],
                rf"(3)  ${sym}$ exiting the target  vs  MIPP measurement",
                meta["label"], f"fig3_{name}_exit_{cfg}.png",
                ref_name=f"MIPP ${sym}$")

    (OUT / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nintegrals over MIPP's measured bins")
    for cfg, d in summary.items():
        print(f"  {mc.CONFIGS[cfg]['short']:16s} ({cfg})")
        for k, v in d.items():
            print(f"      {k:12s} g4numi {v['g4']:7.3f}  ref {v['ref']:7.3f}"
                  f"  ratio {v['ratio']:6.3f}")


if __name__ == "__main__":
    main()
