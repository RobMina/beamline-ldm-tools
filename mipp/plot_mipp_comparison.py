#!/usr/bin/env python
"""Plot g4numi charged-pion yields off the NuMI target against MIPP (PRD 90, 032001)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load(path):
    with open(path) as fh:
        return [{k: (float(v) if v else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def main(csv_name="compare_r0003_fins.csv", out="mipp_vs_g4numi.png"):
    rows = load(HERE / csv_name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5),
                             gridspec_kw={"height_ratios": [2, 1.3]})

    for col, (chg, lab) in enumerate((("pip", r"$\pi^+$"), ("pim", r"$\pi^-$"))):
        ax, axr = axes[0][col], axes[1][col]
        mk = f"n_{chg}"
        good = [r for r in rows if r[mk] is not None]

        # yields vs pz, summed over the measured pT bins
        agg = defaultdict(lambda: np.zeros(4))
        for r in good:
            a = agg[(r["pz_lo"], r["pz_hi"])]
            syst = r[mk] * r[f"d_{chg}_syst"] / 100.0
            stat = r[mk] * r[f"d_{chg}_stat"] / 100.0
            a += [r[mk], r[f"g4_prod_{chg}"], r[f"g4_emerg_{chg}"],
                  stat ** 2 + syst ** 2]
        ks = sorted(agg)
        pzc = np.array([0.5 * (a + b) for a, b in ks])
        pzw = np.array([(b - a) / 2 for a, b in ks])
        m = np.array([agg[k][0] for k in ks])
        me = np.sqrt([agg[k][3] for k in ks])
        gp = np.array([agg[k][1] for k in ks])
        ge = np.array([agg[k][2] for k in ks])

        ax.errorbar(pzc, m, yerr=me, xerr=pzw, fmt="o", color="k", ms=4,
                    lw=1.2, label="MIPP (LE target, measured at target surface)")
        ax.step(np.append(pzc - pzw, pzc[-1] + pzw[-1]), np.append(gp, gp[-1]),
                where="post", color="tab:red",
                label="g4numi ME target, produced in target")
        ax.step(np.append(pzc - pzw, pzc[-1] + pzw[-1]), np.append(ge, ge[-1]),
                where="post", color="tab:blue",
                label=r"g4numi ME target, $\times\,e^{-L/\lambda}$ (emerging)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$N(\pi)/\mathrm{POT}$ per $p_z$ bin")
        ax.set_title(f"{lab}  (summed over MIPP-measured $p_T$ bins)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower left")

        axr.errorbar(pzc, gp / m, xerr=pzw, fmt="s", color="tab:red", ms=4,
                     lw=1.1, label="produced / MIPP")
        axr.errorbar(pzc, ge / m, xerr=pzw, fmt="o", color="tab:blue", ms=4,
                     lw=1.1, label="emerging / MIPP")
        axr.fill_between(pzc, 1 - me / m, 1 + me / m, color="0.7", alpha=0.5,
                         step="mid", label="MIPP uncertainty")
        axr.axhline(1.0, color="k", lw=0.8)
        axr.set_xscale("log")
        axr.set_xlabel(r"$p_z$ (GeV/c)")
        axr.set_ylabel("g4numi / MIPP")
        axr.set_ylim(0.3, 3.0)
        axr.grid(alpha=0.25, which="both")
        axr.legend(fontsize=8, ncol=2)

    fig.suptitle("Charged-pion yield off the NuMI target: g4numi vs MIPP "
                 "(Paley et al., PRD 90, 032001)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / out, dpi=140)
    print("wrote", HERE / out)


def plot_ratio(csv_name="compare_r0003_fins.csv", out="mipp_vs_g4numi_charge_ratio.png"):
    """pi-/pi+ charge ratio: the observable least sensitive to target
    geometry and to the escape model."""
    rows = [r for r in load(HERE / csv_name) if r["ratio"] is not None]
    agg = defaultdict(lambda: np.zeros(5))
    for r in rows:
        a = agg[(r["pz_lo"], r["pz_hi"])]
        a += [r["n_pip"], r["n_pim"], r["g4_prod_pip"], r["g4_prod_pim"],
              (r["n_pim"] * np.hypot(r["d_ratio_stat"], r["d_ratio_syst"]) / 100) ** 2]
    ks = sorted(agg)
    pzc = np.array([0.5 * (a + b) for a, b in ks])
    pzw = np.array([(b - a) / 2 for a, b in ks])
    v = np.array([agg[k] for k in ks])
    r_mipp = v[:, 1] / v[:, 0]
    r_err = np.sqrt(v[:, 4]) / v[:, 0]
    r_g4 = v[:, 3] / v[:, 2]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(pzc, r_mipp, yerr=r_err, xerr=pzw, fmt="o", color="k", ms=4,
                lw=1.2, label="MIPP (LE target)")
    ax.errorbar(pzc, r_g4, xerr=pzw, fmt="s", color="tab:red", ms=4, lw=1.2,
                label="g4numi (ME target, produced in target)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_z$ (GeV/c)")
    ax.set_ylabel(r"$R = N(\pi^-)/N(\pi^+)$")
    ax.set_title("Charged-pion ratio off the NuMI target")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / out, dpi=140)
    print("wrote", HERE / out)


if __name__ == "__main__":
    main(*sys.argv[1:])
    plot_ratio()
