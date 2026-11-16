#!/usr/bin/env python
"""Where along the NuMI beamline does g4numi make its pi0?

MIPP measured hadron production off the *target* only, so it can constrain the
target component of the pi0 flux and nothing else.  This quantifies the split.

A further restriction applies even inside the target: MIPP reports yields only
in its own (pz, pT) bins, which do not cover all of phase space (no pz < 0.3,
nothing in 2 < pz < 4 GeV/c, pT > 0.5 only above pz = 8).  So the *directly*
constrained fraction is smaller still.

Usage:
    python plot_pi0_beamline.py [--configs le_exit me_nom ...]
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

import mipp_common as mc

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("G4NUMI_LDM_OUT", str(HERE / "plots")))


def region_split(z, w, pot, in_graphite):
    """Split production by beamline region.  Inside the target z-window the
    graphite fins are separated from the surrounding vessel/cooling material,
    since only the graphite is what MIPP's target measurement refers to."""
    out = []
    for name, lo, hi in mc.BEAMLINE_REGIONS:
        m = (z >= lo) & (z < hi)
        if name == "target":
            out.append(("target graphite", lo, hi,
                        float(w[m & in_graphite].sum() / pot)))
            out.append(("target vessel / cooling", lo, hi,
                        float(w[m & ~in_graphite].sum() / pot)))
        else:
            out.append((name, lo, hi, float(w[m].sum() / pot)))
    return out


def make_figure(cfg, meta, run, mipp_measured_grid):
    pot, tree = run["pot"], run["trees"]["prod"]
    pi0 = tree["pdg"] == 111
    z, w = tree["z"][pi0], tree["weight"][pi0]
    pz = tree["pz"][pi0]
    pt = np.hypot(tree["px"][pi0], tree["py"][pi0])
    total = w.sum() / pot

    in_tgt = mc.in_target({k: v[pi0] for k, v in tree.items()}, meta["model"])


    # which pi0 land in a bin MIPP actually measured
    bins_pz, bins_pt = mc.mipp_bins()
    ipz = np.digitize(pz, bins_pz) - 1
    ipt = np.digitize(pt, bins_pt) - 1
    inside = ((ipz >= 0) & (ipz < len(bins_pz) - 1)
              & (ipt >= 0) & (ipt < len(bins_pt) - 1))
    in_mipp_bin = np.zeros(len(z), dtype=bool)
    in_mipp_bin[inside] = mipp_measured_grid[ipz[inside], ipt[inside]]

    stats = {
        "config": cfg,
        "pot": pot,
        "total_pi0_per_pot": total,
        "target_pi0_per_pot": float(w[in_tgt].sum() / pot),
        "target_and_mipp_bins_per_pot": float(w[in_tgt & in_mipp_bin].sum() / pot),
        "elsewhere_pi0_per_pot": float(w[~in_tgt].sum() / pot),
        "regions": region_split(z, w, pot, in_tgt),
    }
    stats["target_fraction"] = stats["target_pi0_per_pot"] / total
    stats["mipp_constrained_fraction"] = stats["target_and_mipp_bins_per_pot"] / total

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24,
                          left=0.07, right=0.97, top=0.80, bottom=0.07)

    # (a) target region, fine binning
    ax = fig.add_subplot(gs[0, 0])
    b = np.arange(-450, 520, 2.0)
    ax.hist(z, bins=b, weights=w / pot, histtype="step", color="tab:blue", lw=1.0)
    ax.axvspan(mc.TARGET_Z0, mc.TARGET_Z1 if meta["model"] == "me" else -45.4,
               color="tab:orange", alpha=0.22, label="target graphite")
    ax.set_yscale("log")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\pi^0$ / POT per 2 cm")
    ax.set_title("(a) target region", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) whole beamline
    ax = fig.add_subplot(gs[0, 1])
    b = np.linspace(-500, 76000, 400)
    ax.hist(z, bins=b, weights=w / pot, histtype="step", color="tab:blue", lw=1.0)
    for name, lo, hi in mc.BEAMLINE_REGIONS:
        if lo < -400 or hi > 76000 or name == "target":
            continue
        ax.axvline(lo, color="0.6", lw=0.6, ls=":")
    ax.axvspan(72000, 72800, color="tab:red", alpha=0.18, label="hadron absorber")
    ax.axvspan(mc.TARGET_Z0, 400, color="tab:orange", alpha=0.25,
               label="target + horn 1")
    ax.set_yscale("log")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\pi^0$ / POT per bin")
    ax.set_title("(b) whole beamline", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (c) cumulative
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(z)
    cum = np.cumsum(w[order]) / w.sum()
    ax.plot(z[order], cum, color="tab:blue", lw=1.4)
    ax.axvline(mc.TARGET_Z1, color="tab:orange", ls="--", lw=1.0,
               label="downstream end of target")
    ax.axvline(72000, color="tab:red", ls="--", lw=1.0, label="absorber front")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"cumulative fraction of $\pi^0$ production")
    ax.set_title("(c) cumulative production along z", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # (d) region breakdown
    ax = fig.add_subplot(gs[1, 1])
    regs = [(n, v) for n, _, _, v in stats["regions"] if v > 1e-4]
    names = [n for n, _ in regs][::-1]
    vals = np.array([v for _, v in regs][::-1])
    colors = ["tab:orange" if n.startswith("target") else "0.6" for n in names]
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v * 1.03, i, f"{v:.2f}  ({100*v/total:.1f}%)",
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(right=vals.max() * 6)
    ax.set_xlabel(r"$\pi^0$ / POT")
    ax.set_title("(d) production by beamline region", fontsize=10)
    ax.grid(alpha=0.25, axis="x")

    txt = (f"total $\\pi^0$: {total:.2f}/POT\n"
           f"in target: {stats['target_pi0_per_pot']:.2f}/POT "
           f"({100*stats['target_fraction']:.0f}%)  "
           r"$\leftarrow$ MIPP can speak to this" "\n"
           f"of which inside MIPP's measured bins: "
           f"{stats['target_and_mipp_bins_per_pot']:.2f}/POT "
           f"({100*stats['mipp_constrained_fraction']:.0f}% of all $\\pi^0$)\n"
           f"elsewhere: {stats['elsewhere_pi0_per_pot']:.2f}/POT "
           f"({100*(1-stats['target_fraction']):.0f}%)  "
           r"$\leftarrow$ no MIPP constraint")
    fig.text(0.5, 0.905, txt, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle(r"(4)  Where g4numi makes its $\pi^0$ along the NuMI beamline"
                 f"\n{meta['label']}", fontsize=11.5, y=0.99)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"fig4_pi0_beamline_profile_{cfg}.png", dpi=140)
    plt.close(fig)
    print("wrote", OUT / f"fig4_pi0_beamline_profile_{cfg}.png")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()
    available = [c for c in ("r0003", "me_nom", "me_exit", "le_exit")
                 if (mc.DATA / f"{c}.npz").exists()
                 or (mc.RUNS / c).exists()]
    configs = args.configs or available

    measured = np.isfinite(mc.mipp_grids()["avg"])
    out = {}
    for cfg in configs:
        out[cfg] = make_figure(cfg, mc.CONFIGS[cfg], mc.load_run(cfg), measured)

    (OUT / "beamline_summary.json").write_text(json.dumps(out, indent=2))
    print("\npi0 production split")
    for cfg, s in out.items():
        print(f"  {mc.CONFIGS[cfg]['short']:16s} ({cfg}): total {s['total_pi0_per_pot']:6.2f}"
              f"  target {s['target_pi0_per_pot']:5.2f} ({100*s['target_fraction']:4.1f}%)"
              f"  MIPP-constrained {s['target_and_mipp_bins_per_pot']:5.2f}"
              f" ({100*s['mipp_constrained_fraction']:4.1f}%)")
        for name, _, _, v in s["regions"]:
            if v > 1e-4:
                print(f"        {name:28s} {v:7.3f}/POT  "
                      f"{100*v/s['total_pi0_per_pot']:5.1f}%")


if __name__ == "__main__":
    main()
