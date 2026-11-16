#!/usr/bin/env python
"""Pions g4numi makes *outside* the NuMI target -- the ones MIPP could not see.

MIPP measured charged-pion yields off the target surface, so its constraint
stops at the target.  In g4numi only ~22 % of the beamline pions are born in
the graphite; the rest come from the horns, the decay-pipe wall and the hadron
absorber.  This script asks two things about that majority:

  (1) how its (pz, pT) spectrum compares with the target population MIPP did
      measure, and
  (2) whether pi+ and pi-, which is all MIPP reports, track pi0 well enough to
      be used as a pi0 proxy -- separately inside and outside the target.

Usage:
    python outside_target_pions.py [--run r0003]
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

# The target *assembly* -- graphite fins plus casing and cooling -- is what
# MIPP's "pions emerging from the target" refers to, so the inside/outside
# split is made on the assembly z-window, and the fins are tracked separately.
PZ_BINS = np.concatenate([np.geomspace(0.05, 100.0, 61), [200.0]])
PT_BINS = np.concatenate([np.linspace(0.0, 2.0, 41), [5.0]])
PT_XLIM = (0.0, 2.05)   # the last bin is an overflow, well under 0.1 % of the yield

SPECIES = [(211, r"$\pi^+$", "tab:blue"),
           (-211, r"$\pi^-$", "tab:red"),
           (111, r"$\pi^0$", "tab:green")]


def load(run):
    d = mc.load_run(run)
    tree, pot = d["trees"]["prod"], d["pot"]
    pt = np.hypot(tree["px"], tree["py"])
    z = tree["z"]
    return dict(
        pot=pot,
        pdg=tree["pdg"],
        pz=tree["pz"],
        pt=pt,
        p=np.sqrt(tree["pz"] ** 2 + pt ** 2),
        z=z,
        w=tree["weight"],
        tgen=tree["tgen"],
        fins=mc.in_target(tree, "me"),
        region=(z >= mc.TARGET_Z0) & (z <= mc.TARGET_Z1),
    )


def wq(x, w, q):
    """Weighted quantile."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(q * c[-1], c, x[o]))


def hist(x, w, bins, pot):
    h, _ = np.histogram(x, bins=bins, weights=w)
    return h / pot


def shape(h):
    s = h.sum()
    return h / s if s > 0 else h


def measured_mask(d):
    """True where the pion falls in a (pz, pT) cell MIPP actually reports."""
    grid = np.isfinite(mc.mipp_grids()["avg"])
    bz, bt = mc.mipp_bins()
    iz, it = np.digitize(d["pz"], bz) - 1, np.digitize(d["pt"], bt) - 1
    ok = (iz >= 0) & (iz < len(bz) - 1) & (it >= 0) & (it < len(bt) - 1)
    m = np.zeros(len(iz), dtype=bool)
    m[ok] = grid[iz[ok], it[ok]]
    return m


# ---------------------------------------------------------------- statistics
def summarise(d):
    inb = measured_mask(d)
    out = {}
    for pdg, lab, _ in SPECIES + [(221, "eta", None)]:
        sel = d["pdg"] == pdg
        row = {}
        for tag, m in (("target_fins", sel & d["fins"]),
                       ("target_vessel", sel & d["region"] & ~d["fins"]),
                       ("outside_target", sel & ~d["region"]),
                       ("all", sel)):
            w, pz, pt = d["w"][m], d["pz"][m], d["pt"][m]
            tot = w.sum() / d["pot"]
            row[tag] = {
                "per_pot": tot,
                "mean_pz": float(np.average(pz, weights=w)),
                "median_pz": wq(pz, w, 0.5),
                "mean_pt": float(np.average(pt, weights=w)),
                "median_pt": wq(pt, w, 0.5),
                "frac_pz_gt_4": float(w[pz > 4].sum() / w.sum()),
                "frac_pz_gt_10": float(w[pz > 10].sum() / w.sum()),
                "frac_in_mipp_bins": float(w[inb[m]].sum() / w.sum()),
                "mean_tgen": float(np.average(d["tgen"][m], weights=w)),
            }
        out[str(pdg)] = row
    return out


# ------------------------------------------------------------------- figure 1
def figure_inside_outside(d, run):
    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1], hspace=0.07, wspace=0.26,
                          left=0.065, right=0.985, top=0.855, bottom=0.09)
    axes = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [None, None]]
    axes[1][0] = fig.add_subplot(gs[1, 0], sharex=axes[0][0])
    axes[1][1] = fig.add_subplot(gs[1, 1], sharex=axes[0][1])
    ax2d = fig.add_subplot(gs[:, 2])

    charged = np.isin(d["pdg"], (211, -211))
    pops = [("in the target graphite\n(MIPP measured these)", d["fins"], "tab:orange"),
            ("outside the target\n(MIPP blind)", ~d["region"], "tab:purple")]

    panels = [("pz", PZ_BINS, r"$p_z$ (GeV/c)", True, None),
              ("pt", PT_BINS, r"$p_T$ (GeV/c)", False, PT_XLIM)]

    for col, (var, bins, xlab, logx, xlim) in enumerate(panels):
        ax, axr = axes[0][col], axes[1][col]
        ctr = 0.5 * (bins[1:] + bins[:-1])
        wid = np.diff(bins)
        hs = {}
        for lab, m, c in pops:
            h = hist(d[var][charged & m], d["w"][charged & m], bins, d["pot"])
            hs[lab] = h
            ax.step(ctr, h / wid, where="mid", color=c, lw=1.6, label=lab)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        # MIPP's reach in this variable: pz 0.3-80 GeV/c, pT to 0.5 below
        # pz = 8 GeV/c and to 2.0 above it
        ax.axvspan(*((0.30, 80.0) if var == "pz" else (0.0, 0.5)),
                   color="tab:orange", alpha=0.08, zorder=0)
        ax.set_ylabel(r"$\pi^\pm$ / POT per GeV/c")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        plt.setp(ax.get_xticklabels(), visible=False)

        # two ratios, so that what is drawn above can be read off directly:
        # the solid line is the ratio of the two curves in the panel above, the
        # dashed one divides that out by the 3.5x difference in total yield
        raw = [hs[l] for l, _, _ in pops]
        nrm = [shape(h) for h in raw]
        with np.errstate(invalid="ignore", divide="ignore"):
            r_abs = np.where(raw[0] > 0, raw[1] / raw[0], np.nan)
            r_shp = np.where(nrm[0] > 0, nrm[1] / nrm[0], np.nan)
        axr.step(ctr, r_abs, where="mid", color="k", lw=1.5,
                 label="as plotted above (per POT)")
        axr.step(ctr, r_shp, where="mid", color="tab:purple", lw=1.2, ls="--",
                 label="shape only (each normalised to 1)")
        axr.axhline(1.0, color="0.5", ls="--", lw=0.9)
        axr.set_yscale("log")
        axr.set_ylim(0.02, 30)
        axr.set_xlabel(xlab)
        axr.set_ylabel("outside / target", fontsize=9)
        axr.grid(alpha=0.25)
        if col == 0:
            axr.legend(fontsize=7.5, loc="lower left")
        if xlim:
            ax.set_xlim(*xlim)
            axr.set_xlim(*xlim)

    # right-hand panel: 2D shape ratio outside/target on a coarse (pz, pT) grid
    bz = np.geomspace(0.1, 100, 25)
    bt = np.linspace(0, 1.6, 17)
    grids = []
    for _, m, _ in pops:
        h, _, _ = np.histogram2d(d["pz"][charged & m], d["pt"][charged & m],
                                 bins=[bz, bt], weights=d["w"][charged & m])
        grids.append(h / h.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(grids[0] > 0, grids[1] / grids[0], np.nan)
    pc = ax2d.pcolormesh(bz, bt, ratio.T, cmap="coolwarm",
                         norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=5))
    ax2d.set_xscale("log")
    ax2d.set_xlabel(r"$p_z$ (GeV/c)")
    ax2d.set_ylabel(r"$p_T$ (GeV/c)")
    ax2d.set_title("shape ratio outside / target\n(both normalised to unit area)",
                   fontsize=10)
    fig.colorbar(pc, ax=ax2d, pad=0.02)

    s = summarise(d)
    ch = {k: s[k] for k in ("211", "-211")}
    def avg(tag, key):
        return 0.5 * sum(ch[k][tag][key] for k in ch)
    txt = (r"$\pi^\pm$ born in the target graphite: "
           f"{avg('target_fins','per_pot'):.2f}/POT, "
           r"$\langle p_z\rangle$ = " f"{avg('target_fins','mean_pz'):.1f} GeV/c, "
           r"$\langle p_T\rangle$ = " f"{avg('target_fins','mean_pt'):.3f} GeV/c, "
           f"{100*avg('target_fins','frac_pz_gt_4'):.0f}% above 4 GeV/c\n"
           r"$\pi^\pm$ born outside the target: "
           f"{avg('outside_target','per_pot'):.2f}/POT, "
           r"$\langle p_z\rangle$ = " f"{avg('outside_target','mean_pz'):.1f} GeV/c, "
           r"$\langle p_T\rangle$ = " f"{avg('outside_target','mean_pt'):.3f} GeV/c, "
           f"{100*avg('outside_target','frac_pz_gt_4'):.0f}% above 4 GeV/c")
    fig.text(0.5, 0.925, txt, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle(r"(5)  Charged pions made outside the NuMI target vs. inside it"
                 f"   —   g4numi {run}, ME target", fontsize=12.5, y=0.995)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig5_outside_vs_target_{run}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------- figure 2
# Production regions along the beamline.  Same partition as
# mipp_common.BEAMLINE_REGIONS, with the graphite fins split out of the target
# window because that is the only part MIPP's measurement refers to.
def region_masks(d):
    z, reg = d["z"], d["region"]
    return [
        ("target graphite", d["fins"], "tab:orange"),
        ("target vessel / cooling", reg & ~d["fins"], "tab:brown"),
        ("horn 1", (z >= mc.TARGET_Z1) & (z < 400.0), "tab:blue"),
        ("horn 2 + target hall", (z >= 400.0) & (z < 4500.0), "tab:green"),
        ("decay pipe", (z >= 4500.0) & (z < 72000.0), "tab:red"),
        ("hadron absorber", z >= 72000.0, "tab:purple"),
    ]


def figure_regions(d, run):
    """Same species, different birthplaces: how the spectrum depends on where
    along the beamline the pion was made."""
    regions = region_masks(d)
    panels = [("pz", PZ_BINS, r"$p_z$ (GeV/c)", True, None),
              ("pt", PT_BINS, r"$p_T$ (GeV/c)", False, PT_XLIM)]

    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[2.3, 1, 0.62, 2.3, 1],
                          hspace=0.09, wspace=0.20,
                          left=0.062, right=0.99, top=0.855, bottom=0.06)
    rows = [(0, 1), (3, 4)]

    for pi, (var, bins, xlab, logx, xlim) in enumerate(panels):
        for si, (pdg, slab, _) in enumerate(SPECIES):
            ax = fig.add_subplot(gs[rows[pi][0], si])
            axr = fig.add_subplot(gs[rows[pi][1], si], sharex=ax)
            ctr = 0.5 * (bins[1:] + bins[:-1])
            wid = np.diff(bins)
            ref = None
            for name, m, c in regions:
                s = m & (d["pdg"] == pdg)
                h = hist(d[var][s], d["w"][s], bins, d["pot"])
                ax.step(ctr, h / wid, where="mid", color=c, lw=1.5,
                        label=f"{name}  ({h.sum():.2f}/POT)")
                if ref is None:
                    ref = shape(h)
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(ref > 0, shape(h) / ref, np.nan)
                axr.step(ctr, r, where="mid", color=c, lw=1.2)
            ax.set_yscale("log")
            if logx:
                ax.set_xscale("log")
            ax.set_title(f"{slab},   {'$p_z$' if var == 'pz' else '$p_T$'}",
                         fontsize=11, loc="left", x=0.02)
            ax.legend(fontsize=7, loc="upper right" if var == "pt" else "lower left")
            ax.grid(alpha=0.25)
            plt.setp(ax.get_xticklabels(), visible=False)
            axr.axhline(1.0, color="0.5", ls="--", lw=0.9)
            axr.set_yscale("log")
            axr.set_ylim(0.03, 30)
            axr.set_xlabel(xlab)
            axr.grid(alpha=0.25)
            if si == 0:
                ax.set_ylabel("mesons / POT per GeV/c")
                axr.set_ylabel("shape /\ntarget graphite", fontsize=8.5)
            if xlim:
                ax.set_xlim(*xlim)
                axr.set_xlim(*xlim)

    rows = []
    for name, m, _ in regions:
        s = m & np.isin(d["pdg"], (211, -211, 111))
        w = d["w"][s]
        rows.append(f"{name}: {w.sum()/d['pot']:.2f} $\\pi$/POT, "
                    f"$\\langle p_z\\rangle$ = {np.average(d['pz'][s], weights=w):.2f}, "
                    f"$\\langle p_T\\rangle$ = {np.average(d['pt'][s], weights=w):.3f} GeV/c")
    fig.text(0.5, 0.955, "   |   ".join(rows[:3]) + "\n" + "   |   ".join(rows[3:]),
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle("(6)  Pion spectra by production location along the NuMI beamline"
                 f"   —   g4numi {run}, ME target", fontsize=12.5, y=0.997)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig6_regions_{run}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------- figure 3
def figure_charges(d, run):
    """Is (pi+ + pi-)/2 a good stand-in for pi0?  Inside and outside the target."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.07,
                                              wspace=0.26))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.865, bottom=0.085)

    cases = [(r"$\pi$ born in the target graphite", d["fins"]),
             (r"$\pi$ born outside the target", ~d["region"])]
    panels = [("pz", PZ_BINS, r"$p_z$ (GeV/c)", True, None),
              ("pt", PT_BINS, r"$p_T$ (GeV/c)", False, PT_XLIM)]

    for ci, (cname, cmask) in enumerate(cases):
        for pi, (var, bins, xlab, logx, xlim) in enumerate(panels):
            col = 2 * ci + pi
            ax, axr = axes[0][col], axes[1][col]
            ctr = 0.5 * (bins[1:] + bins[:-1])
            wid = np.diff(bins)
            hs = {}
            for pdg, lab, c in SPECIES:
                m = cmask & (d["pdg"] == pdg)
                h = hist(d[var][m], d["w"][m], bins, d["pot"])
                hs[pdg] = h
                ax.step(ctr, h / wid, where="mid", color=c, lw=1.5, label=lab)
            ax.set_yscale("log")
            if logx:
                ax.set_xscale("log")
                axr.set_xscale("log")
            if col == 0:
                ax.set_ylabel(r"mesons / POT per GeV/c")
            if pi == 0:
                ax.set_title(cname, fontsize=11.5, loc="left", x=0.02)
            ax.legend(fontsize=8.5)
            ax.grid(alpha=0.25)

            chavg = 0.5 * (hs[211] + hs[-211])
            with np.errstate(invalid="ignore", divide="ignore"):
                yield_ratio = np.where(chavg > 0, hs[111] / chavg, np.nan)
                shape_ratio = np.where(shape(chavg) > 0,
                                       shape(hs[111]) / shape(chavg), np.nan)
                charge_ratio = np.where(hs[211] > 0, hs[-211] / hs[211], np.nan)
            axr.step(ctr, yield_ratio, where="mid", color="k", lw=1.4,
                     label=r"$\pi^0$ / $\langle\pi^\pm\rangle$")
            axr.step(ctr, shape_ratio, where="mid", color="tab:green", lw=1.2,
                     ls="--", label="same, shape only")
            axr.step(ctr, charge_ratio, where="mid", color="tab:red", lw=1.0,
                     ls=":", label=r"$\pi^-/\pi^+$")
            axr.axhline(1.0, color="0.5", ls="--", lw=0.9)
            axr.set_ylim(0.4, 1.8)
            axr.set_xlabel(xlab)
            if col == 0:
                axr.set_ylabel("ratio", fontsize=9)
                axr.legend(fontsize=7.5, ncol=1, loc="lower left")
            axr.grid(alpha=0.25)
            if xlim:
                ax.set_xlim(*xlim)
                axr.set_xlim(*xlim)

    s = summarise(d)
    def line(tag):
        p0, pp, pm = s["111"][tag], s["211"][tag], s["-211"][tag]
        avg = 0.5 * (pp["per_pot"] + pm["per_pot"])
        return (f"{p0['per_pot']:.2f} / {avg:.2f} = {p0['per_pot']/avg:.3f}"
                f"   ($\\langle p_z\\rangle$ {p0['mean_pz']:.1f} vs "
                f"{0.5*(pp['mean_pz']+pm['mean_pz']):.1f} GeV/c,"
                f"  $\\langle p_T\\rangle$ {p0['mean_pt']:.3f} vs "
                f"{0.5*(pp['mean_pt']+pm['mean_pt']):.3f} GeV/c)")
    fig.text(0.5, 0.965,
             r"$\pi^0$ / $\langle\pi^\pm\rangle$ — target graphite: " + line("target_fins")
             + "\n" + r"$\pi^0$ / $\langle\pi^\pm\rangle$ — outside the target: "
             + line("outside_target"),
             ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle(r"(9)  Are MIPP's $\pi^\pm$ kinematics a proxy for $\pi^0$?"
                 f"   —   g4numi {run}, ME target", fontsize=12.5, y=0.998)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig9_charge_proxy_{run}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r0003")
    args = ap.parse_args()

    d = load(args.run)
    s = summarise(d)
    (HERE / f"outside_target_summary_{args.run}.json").write_text(json.dumps(s, indent=2))
    figure_inside_outside(d, args.run)
    figure_regions(d, args.run)
    figure_charges(d, args.run)

    hdr = f"{'':10s}" + "".join(f"{t:>16s}" for t in
                                ("per POT", "<pz>", "med pz", "<pT>", "med pT",
                                 ">4 GeV/c", "in MIPP bins", "<tgen>"))
    for tag in ("target_fins", "target_vessel", "outside_target", "all"):
        print(f"\n--- {tag}\n{hdr}")
        for pdg, lab in ((211, "pi+"), (-211, "pi-"), (111, "pi0"), (221, "eta")):
            r = s[str(pdg)][tag]
            print(f"{lab:10s}"
                  + "".join(f"{v:>16.4g}" for v in
                            (r["per_pot"], r["mean_pz"], r["median_pz"],
                             r["mean_pt"], r["median_pt"], r["frac_pz_gt_4"],
                             r["frac_in_mipp_bins"], r["mean_tgen"])))


if __name__ == "__main__":
    main()
