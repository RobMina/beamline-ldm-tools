#!/usr/bin/env python
"""A MIPP-derived (p_z, p_T) correction for g4numi's in-target pi0 production.

The correction is the data/MC ratio of the *same observable* -- charged pions
crossing the outer surface of the target --

    C(p_z, p_T)  =  MIPP <pi+-> exiting the target
                    ---------------------------------
                    g4numi <pi+-> exiting the target

evaluated like-for-like: the g4numi side is the `le_exit` run (LE target, MIPP's
0.11 mm pencil beam, truth `tgtexit` tree), so the target geometry, the beam spot
and the definition of "exiting" all match what MIPP reports.  C is then applied
to the pi0 *produced in the target* of the ME production run.

Why this is better than converting MIPP into a production rate first (which is
what sections 7 and 14 do): that route needs a per-bin escape factor, and the
g4numi exit/production ratio is not a clean survival probability -- it exceeds 1
below 1 GeV/c because the exit tree counts mesons made anywhere inside the target
can while the production selection is the graphite fins only, and it falls to 0.5
at 80 GeV/c through a mix of absorption and (p_z, p_T) migration.  Taking the
ratio of two exit fluxes cancels all of that: absorption, migration and the
volume definition are common to numerator and denominator.

What it assumes, in exchange:

  * that g4numi's absorption and migration inside the target are right, so that a
    residual data/MC difference in the exiting flux is a *production* difference;
  * that whatever g4numi gets wrong in charged-pion production at a given
    (p_z, p_T) it gets equally wrong for pi0 there.  Section 11 is the internal
    evidence for that (pi0/<pi+-> = 1.02 in the target, TV <= 0.07 in shape) --
    but it is an assumption, and it is exactly the same one the reference analysis's chain makes
    when it uses MIPP's charged average as a pi0 source;
  * that a production-level modelling error measured on the LE target carries
    over to the ME target.  It is a property of the physics list, not of the
    geometry, so it should; the ME target's extra re-interaction makes the mix of
    interaction types slightly different.

Usage:
    python mipp_pi0_correction.py [--run r0003] [--mchi 0.012]
"""
from __future__ import annotations

import os

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mipp_common as mc

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("G4NUMI_LDM_OUT", str(HERE / "plots")))

# The part of MIPP's systematic that is common to essentially every bin
# (4 % bin migration + 2 % efficiency + 1 % noise, section 1).  It does not
# average down when bins are summed, so it is tracked separately.
MIPP_CORRELATED_SYST = 0.046
RATIO_GUARD = (0.0, 10.0)     # same guard as NOvA_pion_dist.ipynb


def mipp_avg_with_errors():
    """MIPP's charged average on its own grid, split into correlated and
    uncorrelated uncertainty."""
    bz, bt = mc.mipp_bins()
    shape = (len(bz) - 1, len(bt) - 1)
    val = np.full(shape, np.nan)
    unc = np.full(shape, np.nan)      # uncorrelated part, absolute
    for r in mc.mipp_rows():
        if r["n_pip"] is None or r["n_pim"] is None:
            continue
        i = int(np.searchsorted(bz, r["pz_lo"], side="right") - 1)
        j = int(np.searchsorted(bt, r["pt_lo"], side="right") - 1)
        v = 0.5 * (r["n_pip"] + r["n_pim"])
        # stat is independent between the two charges; the systematic beyond the
        # common floor is treated as uncorrelated bin to bin
        stat = 0.5 * np.hypot(r["n_pip"] * r["d_pip_stat"] / 100.0,
                              r["n_pim"] * r["d_pim_stat"] / 100.0)
        syst = max(r["d_pip_syst"], r["d_pim_syst"]) / 100.0
        extra = np.sqrt(max(syst**2 - MIPP_CORRELATED_SYST**2, 0.0))
        val[i][j] = v
        unc[i][j] = np.hypot(stat, v * extra)
    return val, unc


def g4numi_exit_with_errors(config="le_exit"):
    """g4numi charged-pion flux crossing the target surface, on MIPP's grid,
    with the MC statistical error from the importance weights."""
    d = mc.load_run(config)
    pot, ex = d["pot"], d["trees"]["exit"]
    bz, bt = mc.mipp_bins()
    val = np.zeros((len(bz) - 1, len(bt) - 1))
    var = np.zeros_like(val)
    for pdg in (211, -211):
        pz, pt, w = mc.select(ex, pdg)
        h, _, _ = np.histogram2d(pz, pt, bins=[bz, bt], weights=w)
        h2, _, _ = np.histogram2d(pz, pt, bins=[bz, bt], weights=w**2)
        val += 0.5 * h / pot
        var += 0.25 * h2 / pot**2
    return val, np.sqrt(var)


def build_correction(config="le_exit"):
    m, dm = mipp_avg_with_errors()
    g, dg = g4numi_exit_with_errors(config)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where((g > 0) & np.isfinite(m), m / g, np.nan)
        # uncorrelated relative error on C, plus the flat correlated MIPP piece
        rel = np.sqrt(np.where(m > 0, (dm / m) ** 2, 0)
                      + np.where(g > 0, (dg / g) ** 2, 0))
        dC = C * rel
    ok = np.isfinite(C) & (C > RATIO_GUARD[0]) & (C < RATIO_GUARD[1])
    return dict(C=C, dC=dC, ok=ok, mipp=m, g4=g, dmipp=dm, dg4=dg)


def apply_to_events(pz, pt, corr, mode="unity"):
    """Per-event correction factor.

    `mode` says what to do where MIPP reports nothing:
      'unity'   -- weight 1 (the guard NOvA_pion_dist.ipynb uses)
      'nearest' -- carry the nearest measured p_z bin at the same p_T across
    Returns (weight, covered) with covered True where MIPP actually measured.
    """
    bz, bt = mc.mipp_bins()
    C, ok = corr["C"], corr["ok"]
    iz, it = np.digitize(pz, bz) - 1, np.digitize(pt, bt) - 1
    inside = ((iz >= 0) & (iz < len(bz) - 1) & (it >= 0) & (it < len(bt) - 1))
    izc, itc = np.clip(iz, 0, len(bz) - 2), np.clip(it, 0, len(bt) - 2)

    grid = np.where(ok, C, np.nan)
    if mode == "nearest":
        filled = grid.copy()
        for j in range(grid.shape[1]):
            col = grid[:, j]
            good = np.flatnonzero(np.isfinite(col))
            if len(good) == 0:
                continue
            idx = np.arange(len(col))
            filled[:, j] = col[good[np.abs(idx[:, None] - good[None, :]).argmin(1)]]
        grid = filled

    w = np.ones(len(pz))
    covered = np.zeros(len(pz), dtype=bool)
    v = grid[izc, itc]
    use = inside & np.isfinite(v)
    w[use] = v[use]
    covered[inside] = ok[izc[inside], itc[inside]]
    return w, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r0003", help="ME production run to correct")
    ap.add_argument("--config", default="le_exit", help="g4numi run the ratio comes from")
    ap.add_argument("--mchi", type=float, default=0.012)
    ap.add_argument("--nsample", type=int, default=4)
    args = ap.parse_args()

    corr = build_correction(args.config)
    bz, bt = mc.mipp_bins()

    # ---- write the correction map -----------------------------------------
    with open(HERE / "mipp_pi0_correction.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["pz_lo", "pz_hi", "pt_lo", "pt_hi", "mipp_avg", "mipp_unc_uncorr",
                       "g4numi_exit", "g4numi_mc_stat", "correction", "correction_unc"])
        for i in range(len(bz) - 1):
            for j in range(len(bt) - 1):
                if not corr["ok"][i, j]:
                    continue
                wcsv.writerow([bz[i], bz[i+1], bt[j], bt[j+1],
                               f"{corr['mipp'][i,j]:.6g}", f"{corr['dmipp'][i,j]:.4g}",
                               f"{corr['g4'][i,j]:.6g}", f"{corr['dg4'][i,j]:.4g}",
                               f"{corr['C'][i,j]:.4f}", f"{corr['dC'][i,j]:.4f}"])
    print("wrote", HERE / "mipp_pi0_correction.csv",
          f"({int(corr['ok'].sum())} bins)")

    # ---- apply to the in-target pi0 of the ME production run --------------
    r = mc.load_run(args.run)
    t, pot = r["trees"]["prod"], r["pot"]
    intgt = mc.in_target(t, "me")
    sel = (t["pdg"] == 111) & intgt
    pz = t["pz"][sel]
    pt = np.hypot(t["px"][sel], t["py"][sel])
    w = t["weight"][sel]

    res = {"run": args.run, "ratio_from": args.config,
           "n_bins": int(corr["ok"].sum()),
           "correction_weighted_mean": float(
               np.nansum(np.where(corr["ok"], corr["C"] * corr["g4"], 0))
               / np.nansum(np.where(corr["ok"], corr["g4"], 0)))}

    for mode in ("unity", "nearest"):
        cw, cov = apply_to_events(pz, pt, corr, mode)
        row = {"coverage_frac_of_target_pi0": float(w[cov].sum() / w.sum())}
        for lab, m in (("total", np.ones(len(pz), bool)),
                       ("in_mipp_bins", cov),
                       ("pz_gt_4", pz > 4.0),
                       ("pz_gt_10", pz > 10.0)):
            row[lab] = {"before": float(w[m].sum() / pot),
                        "after": float((w[m] * cw[m]).sum() / pot)}
            row[lab]["factor"] = row[lab]["after"] / row[lab]["before"]
        res[mode] = row

    # ---- and in the quantity that matters: chi crossing the ND ------------
    import dm_angular_spread as D
    a = D.analyse(args.run, args.mchi, args.nsample)
    tsel = t["pdg"] == 111
    pz_a = np.tile(t["pz"][tsel], args.nsample)
    pt_a = np.tile(np.hypot(t["px"][tsel], t["py"][tsel]), args.nsample)
    cw_a, _ = apply_to_events(pz_a, pt_a, corr, "unity")
    cw_a = np.where(a["reg"] == 0, cw_a, 1.0)     # target component only
    hit, ww = a["hit"], a["w"]
    tgt = a["reg"] == 0
    res["chi_at_nd_per_pot"] = {
        "target, uncorrected": float(2 * ww[hit & tgt].sum() / a["pot"]),
        "target, MIPP-corrected": float(2 * (ww * cw_a)[hit & tgt].sum() / a["pot"]),
        "whole beamline, uncorrected": float(2 * ww[hit].sum() / a["pot"]),
        "whole beamline, target part MIPP-corrected":
            float(2 * (ww * cw_a)[hit].sum() / a["pot"]),
    }
    c = res["chi_at_nd_per_pot"]
    res["chi_at_nd_per_pot"]["target factor"] = (
        c["target, MIPP-corrected"] / c["target, uncorrected"])
    res["chi_at_nd_per_pot"]["beamline factor"] = (
        c["whole beamline, target part MIPP-corrected"] / c["whole beamline, uncorrected"])

    # ---- uncertainty on the corrected target pi0 --------------------------
    # the correlated 4.6 % floor passes straight through; the uncorrelated part
    # is propagated bin by bin, weighted by how much pi0 each bin carries
    cw, cov = apply_to_events(pz, pt, corr, "unity")
    for lab, m in (("total", np.ones(len(pz), bool)), ("pz_gt_4", pz > 4.0)):
        iz, it = np.digitize(pz[m], bz) - 1, np.digitize(pt[m], bt) - 1
        good = cov[m]
        contrib = np.zeros_like(corr["C"])
        np.add.at(contrib, (np.clip(iz, 0, len(bz)-2)[good],
                            np.clip(it, 0, len(bt)-2)[good]),
                  (w[m] * cw[m])[good])
        with np.errstate(invalid="ignore", divide="ignore"):
            relu = np.where(corr["ok"], corr["dC"] / corr["C"], 0.0)
        unc_uncorr = np.sqrt(np.nansum((contrib * relu) ** 2)) / pot
        corrected = (w[m] * cw[m]).sum() / pot
        frac_cov = contrib.sum() / (w[m] * cw[m]).sum()
        res[f"uncertainty_{lab}"] = {
            "corrected_per_pot": float(corrected),
            "uncorrelated": float(unc_uncorr),
            "correlated_mipp_floor": float(MIPP_CORRELATED_SYST * corrected * frac_cov),
            "total": float(np.hypot(unc_uncorr,
                                    MIPP_CORRELATED_SYST * corrected * frac_cov)),
        }

    (HERE / "mipp_pi0_correction.json").write_text(json.dumps(res, indent=2))
    figure(corr, pz, pt, w, pot, res, args)
    print(json.dumps(res, indent=2))


def figure(corr, pz, pt, w, pot, res, args):
    bz, bt = mc.mipp_bins()
    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28,
                          left=0.065, right=0.97, top=0.82, bottom=0.08)

    # (a) the correction map
    ax = fig.add_subplot(gs[0, 0])
    C = np.where(corr["ok"], corr["C"], np.nan)
    pc = ax.pcolormesh(np.arange(len(bz)), np.arange(len(bt)), C.T,
                       cmap="coolwarm",
                       norm=matplotlib.colors.LogNorm(vmin=0.4, vmax=2.5))
    ax.set_xticks(np.arange(len(bz))[::3] + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in bz[::3]], fontsize=7.5)
    ax.set_yticks(np.arange(len(bt)) + 0.5)
    ax.set_yticklabels([f"{v:g}" for v in bt], fontsize=7.5)
    ax.set_xlabel(r"$p_z$ (GeV/c), bin edges")
    ax.set_ylabel(r"$p_T$ (GeV/c)")
    ax.set_title("(a) C = MIPP / g4numi, exiting the target", fontsize=10)
    fig.colorbar(pc, ax=ax, pad=0.02)

    # (b) C projected on pz
    ax = fig.add_subplot(gs[0, 1])
    num = np.nansum(np.where(corr["ok"], corr["mipp"], 0), axis=1)
    den = np.nansum(np.where(corr["ok"], corr["g4"], 0), axis=1)
    ctr = np.sqrt(bz[:-1] * bz[1:])
    m = den > 0
    ax.step(ctr[m], (num / den)[m], where="mid", color="k", lw=1.6)
    ax.axhline(1.0, color="0.5", ls="--")
    ax.axvline(4.0, color="tab:red", ls=":", lw=1.2)
    ax.text(4.3, 2.15, "signal region\n$p_z$ > 4 GeV/c", color="tab:red", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_z$ (GeV/c)")
    ax.set_ylabel("C")
    ax.set_ylim(0, 2.6)
    ax.set_title("(b) C projected on $p_z$", fontsize=10)
    ax.grid(alpha=0.25)

    # (c) C projected on pT
    ax = fig.add_subplot(gs[0, 2])
    num = np.nansum(np.where(corr["ok"], corr["mipp"], 0), axis=0)
    den = np.nansum(np.where(corr["ok"], corr["g4"], 0), axis=0)
    ctr = 0.5 * (bt[:-1] + bt[1:])
    m = den > 0
    ax.step(ctr[m], (num / den)[m], where="mid", color="k", lw=1.6)
    ax.axhline(1.0, color="0.5", ls="--")
    ax.set_xlabel(r"$p_T$ (GeV/c)")
    ax.set_ylabel("C")
    ax.set_ylim(0, 2.6)
    ax.set_title("(c) C projected on $p_T$", fontsize=10)
    ax.grid(alpha=0.25)

    # (d)(e) the in-target pi0 spectrum before and after
    cw, cov = apply_to_events(pz, pt, corr, "unity")
    for k, (var, bins, xlab, logx) in enumerate(
            [(pz, np.geomspace(0.1, 100, 45), r"$p_z$ (GeV/c)", True),
             (pt, np.linspace(0, 2, 33), r"$p_T$ (GeV/c)", False)]):
        ax = fig.add_subplot(gs[1, k])
        c0, _ = np.histogram(var, bins, weights=w)
        c1, _ = np.histogram(var, bins, weights=w * cw)
        x = 0.5 * (bins[1:] + bins[:-1])
        d = np.diff(bins)
        ax.step(x, c0 / pot / d, where="mid", color="tab:blue", lw=1.5,
                label=f"g4numi ME target $\\pi^0$ "
                      f"({res['unity']['total']['before']:.2f}/POT total)")
        ax.step(x, c1 / pot / d, where="mid", color="tab:red", lw=1.5,
                label=f"after the MIPP correction "
                      f"({res['unity']['total']['after']:.2f}/POT total)")
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\pi^0$ / POT per GeV/c")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_title(f"({'de'[k]}) in-target $\\pi^0$, before and after", fontsize=10)

    # (f) the numbers
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    u = res["uncertainty_total"]
    uf = res["uncertainty_pz_gt_4"]
    c = res["chi_at_nd_per_pot"]
    lines = [
        r"$\bf{in-target\ \pi^0\ per\ POT}$",
        f"   total          {res['unity']['total']['before']:.3f}  →  "
        f"{res['unity']['total']['after']:.3f}   (×{res['unity']['total']['factor']:.3f})",
        f"   inside MIPP bins {res['unity']['in_mipp_bins']['before']:.3f}  →  "
        f"{res['unity']['in_mipp_bins']['after']:.3f}   (×{res['unity']['in_mipp_bins']['factor']:.3f})",
        f"   $p_z$ > 4 GeV/c    {res['unity']['pz_gt_4']['before']:.3f}  →  "
        f"{res['unity']['pz_gt_4']['after']:.3f}   (×{res['unity']['pz_gt_4']['factor']:.3f})",
        f"   $p_z$ > 10 GeV/c   {res['unity']['pz_gt_10']['before']:.3f}  →  "
        f"{res['unity']['pz_gt_10']['after']:.3f}   (×{res['unity']['pz_gt_10']['factor']:.3f})",
        "",
        r"$\bf{uncertainty\ on\ the\ corrected\ rate}$",
        f"   total     ±{100*u['uncorrelated']/u['corrected_per_pot']:.1f} % uncorr. "
        f"⊕ {100*u['correlated_mipp_floor']/u['corrected_per_pot']:.1f} % MIPP floor "
        f"= ±{100*u['total']/u['corrected_per_pot']:.1f} %",
        f"   $p_z$ > 4    ±{100*uf['uncorrelated']/uf['corrected_per_pot']:.1f} % ⊕ "
        f"{100*uf['correlated_mipp_floor']/uf['corrected_per_pot']:.1f} % "
        f"= ±{100*uf['total']/uf['corrected_per_pot']:.1f} %",
        "",
        r"$\bf{\chi\ crossing\ the\ ND\ per\ POT}$" f"  ($m_\\chi$ = {1e3*args.mchi:.0f} MeV)",
        f"   target only    {c['target, uncorrected']:.3e}  →  "
        f"{c['target, MIPP-corrected']:.3e}   (×{c['target factor']:.3f})",
        f"   whole beamline {c['whole beamline, uncorrected']:.3e}  →  "
        f"{c['whole beamline, target part MIPP-corrected']:.3e}   "
        f"(×{c['beamline factor']:.3f})",
        "",
        f"coverage: {100*res['unity']['coverage_frac_of_target_pi0']:.0f} % of the "
        "in-target $\\pi^0$ weight",
        f"falls in one of the {res['n_bins']} bins MIPP reports;",
        "the rest is left at weight 1.  Filling the gaps by",
        "carrying the nearest measured $p_z$ bin instead gives",
        f"×{res['nearest']['total']['factor']:.3f} on the total and "
        f"×{res['nearest']['pz_gt_4']['factor']:.3f} above 4 GeV/c.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
            family="DejaVu Sans", transform=ax.transAxes)

    fig.suptitle("(11)  MIPP-derived correction to g4numi's in-target $\\pi^0$ production\n"
                 r"$C(p_z,p_T)$ = MIPP $\langle\pi^\pm\rangle$ / g4numi $\langle\pi^\pm\rangle$, "
                 "both exiting the target, LE geometry + MIPP beam spot",
                 fontsize=12, y=0.975)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig11_mipp_pi0_correction_{args.run}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
