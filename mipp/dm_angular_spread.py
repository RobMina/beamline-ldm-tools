#!/usr/bin/env python
"""How much wider is the DM cone once pi0 from the whole beamline are included?

BMPT/BdNMC in the reference analysis's chain only ever saw pi0 made in -- and decaying in -- the
graphite target, all of them ~1 km upstream of the ND and all within a few
milliradians of the beam axis.  g4numi says that population is only ~23 % of
the beamline pi0: the rest are born in horn 1, the target hall, the decay-pipe
wall and the hadron absorber, some of them only ~280 m from the ND.

For each pi0 this script simulates pi0 -> gamma A', A' -> chi chibar with the
BdNMC mass convention m_A' = 3 m_chi, and reports, per production region:

  * theta_ND -- the angle between the chi momentum and the line joining its
    production point to the ND centre.  This is the quantity that has to be
    small for the chi to reach the detector, and it is what "fuzzier" means.
  * the half-angle the ND itself subtends from that production point, which
    grows as the source moves downstream and partly pays the broadening back.
  * the geometric acceptance -- the fraction of chi whose straight-line
    trajectory actually crosses the ND box -- which is the combination of the
    two that the signal rate cares about.

Usage:
    python dm_angular_spread.py [--run r0003] [--mchi 0.012]
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

M_PI0 = 0.1349768   # GeV
M_ETA = 0.547862

# Detector placement in the target-centred beam frame: from the configuration ($LDM_DETECTOR_CONFIG, see
# g4numi_ldm/detector.py); the experiment's values are not in this repository.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4numi_ldm import detector as _det  # noqa: E402
_cfg = _det.load()
X0_NOVA = np.array(_cfg["frame"]["origin_m"], dtype=float)       # front-face centre, m
ALPHA, BETA = float(_cfg["frame"]["pitch_rad"]), float(_cfg["frame"]["yaw_rad"])  # pitch, yaw (rad)
DET_HALF_W = float(_cfg["detector"]["half_width_m"])            # m
DET_LEN = float(_cfg["detector"]["length_m"])                   # m
# g4numi MC z = 0 sits 0.8165 m upstream of the target centre that the BdNMC /
# NOvASoft beam frame uses as its origin .
G4_TO_BEAM_Z = 0.8165

_CA, _SA = np.cos(ALPHA), np.sin(ALPHA)
_CB, _SB = np.cos(BETA), np.sin(BETA)
R_NOVA = np.array([[_CB, -_SB * _SA, _SB * _CA],
                   [0.0, _CA, _SA],
                   [-_SB, -_CB * _SA, _CB * _CA]])
DET_CENTRE = X0_NOVA + R_NOVA @ np.array([0.0, 0.0, DET_LEN / 2])

REGIONS = [
    ("target graphite", "tab:orange"),
    ("target vessel / horn 1", "tab:blue"),
    ("horn 2 + target hall", "tab:green"),
    ("decay pipe", "tab:red"),
    ("hadron absorber", "tab:purple"),
]


def region_of(z_cm, in_fins):
    """Coarse production-region label per pi0, as an index into REGIONS."""
    r = np.full(len(z_cm), -1, dtype=np.int8)
    r[(z_cm >= mc.TARGET_Z0) & (z_cm < 400.0)] = 1      # vessel + horn 1
    r[in_fins] = 0
    r[(z_cm >= 400.0) & (z_cm < 4500.0)] = 2
    r[(z_cm >= 4500.0) & (z_cm < 72000.0)] = 3
    r[z_cm >= 72000.0] = 4
    r[z_cm < mc.TARGET_Z0] = 1                          # upstream/baffle, tiny
    return r


def two_body(m0, m1, m2, n, rng):
    """Isotropic 2-body decay in the parent rest frame -> (E1, p1 vector)."""
    p = np.sqrt(np.maximum((m0**2 - (m1 + m2)**2) * (m0**2 - (m1 - m2)**2), 0.0)) / (2 * m0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t**2)
    d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    return np.full(n, np.sqrt(p**2 + m1**2)), p * d


def boost(E, p, beta_vec):
    """Boost a 4-vector (E, p) by beta_vec (velocity of the new frame's origin
    in the target frame, i.e. the usual 'from rest frame to lab' sense)."""
    b2 = np.einsum("ij,ij->i", beta_vec, beta_vec)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.einsum("ij,ij->i", beta_vec, p)
    # gamma^2/(gamma+1) form avoids the 0/0 at beta -> 0
    fac = (gamma - 1.0) / np.where(b2 > 0, b2, 1.0)
    p_lab = p + (fac * bp)[:, None] * beta_vec + (gamma * E)[:, None] * beta_vec
    E_lab = gamma * (E + bp)
    return E_lab, p_lab


def decay_to_chi(E, px, py, pz, m_parent, m_ap, m_chi, rng):
    """parent -> gamma A', A' -> chi chibar.  Returns the lab (E, p) of one chi."""
    n = len(E)
    # parent rest frame: gamma + A'
    E_ap, p_ap = two_body(m_parent, m_ap, 0.0, n, rng)
    # A' rest frame: chi + chibar
    E_chi0, p_chi0 = two_body(m_ap, m_chi, m_chi, n, rng)
    # A' rest frame -> parent rest frame
    beta_ap = p_ap / E_ap[:, None]
    E_chi1, p_chi1 = boost(E_chi0, p_chi0, beta_ap)
    # parent rest frame -> lab
    p_par = np.stack([px, py, pz], axis=1)
    beta_par = p_par / E[:, None]
    return boost(E_chi1, p_chi1, beta_par)


def hits_detector(origin_m, direc):
    """Slab test: does the ray from origin_m along direc cross the ND box?"""
    o = (origin_m - X0_NOVA) @ R_NOVA          # R^T x, written as a row-vector product
    d = direc @ R_NOVA
    lo = np.array([-DET_HALF_W, -DET_HALF_W, 0.0])
    hi = np.array([DET_HALF_W, DET_HALF_W, DET_LEN])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return (tmax >= np.maximum(tmin, 0.0))


def analyse(run, m_chi, n_sample, seed=7):
    m_ap = 3.0 * m_chi
    if m_ap >= M_PI0:
        raise SystemExit(f"m_A' = {m_ap:.3f} GeV is above m_pi0; pi0 decay is closed")
    rng = np.random.default_rng(seed)

    d = mc.load_run(run)
    t, pot = d["trees"]["prod"], d["pot"]
    sel = t["pdg"] == 111
    E = np.sqrt(t["px"][sel]**2 + t["py"][sel]**2 + t["pz"][sel]**2 + M_PI0**2)
    px, py, pz = t["px"][sel], t["py"][sel], t["pz"][sel]
    w = t["weight"][sel]
    pos = np.stack([t["x"][sel], t["y"][sel], t["z"][sel]], axis=1) / 100.0
    pos[:, 2] += G4_TO_BEAM_Z
    reg = region_of(t["z"][sel], mc.in_target({k: v[sel] for k, v in t.items()}, "me"))

    # line of sight from each production point to the ND centre
    los = DET_CENTRE - pos
    dist = np.linalg.norm(los, axis=1)
    los /= dist[:, None]
    # half-angle the ND subtends from there (equal-area disc of the 3.9x3.9 m face)
    subtend = np.arctan(np.sqrt(DET_HALF_W**2 * 4 / np.pi) / dist)

    theta_nd, theta_beam, hit, echi, wrep, regrep = [], [], [], [], [], []
    for _ in range(n_sample):
        Ec, pc = decay_to_chi(E, px, py, pz, M_PI0, m_ap, m_chi, rng)
        u = pc / np.linalg.norm(pc, axis=1)[:, None]
        theta_nd.append(np.arccos(np.clip(np.einsum("ij,ij->i", u, los), -1, 1)))
        theta_beam.append(np.arccos(np.clip(u[:, 2], -1, 1)))
        hit.append(hits_detector(pos, u))
        echi.append(Ec)
        wrep.append(w)
        regrep.append(reg)
    out = dict(
        pot=pot, m_chi=m_chi, m_ap=m_ap, n_sample=n_sample,
        theta_nd=np.concatenate(theta_nd), theta_beam=np.concatenate(theta_beam),
        hit=np.concatenate(hit), E=np.concatenate(echi),
        w=np.concatenate(wrep) / n_sample, reg=np.concatenate(regrep),
        subtend=np.tile(subtend, n_sample), dist=np.tile(dist, n_sample),
        pi0_w=w, pi0_reg=reg, pi0_dist=dist, pi0_subtend=subtend,
    )
    return out


def summarise(a):
    rows = {}
    for i, (name, _) in enumerate(REGIONS):
        m = a["reg"] == i
        if not m.any():
            continue
        w, th = a["w"][m], a["theta_nd"][m]
        pm = a["pi0_reg"] == i
        rows[name] = {
            "pi0_per_pot": float(a["pi0_w"][pm].sum() / a["pot"]),
            "median_dist_m": float(np.median(a["pi0_dist"][pm])),
            "median_subtend_mrad": float(1e3 * np.median(a["pi0_subtend"][pm])),
            "median_theta_nd_mrad": float(1e3 * np.median(th)),
            "frac_theta_lt_10mrad": float(w[th < 0.010].sum() / w.sum()),
            "acceptance": float(a["w"][m & a["hit"]].sum() / w.sum()),
            "chi_to_nd_per_pot": float(2 * a["w"][m & a["hit"]].sum() / a["pot"]),
            "mean_E_hit_GeV": (float(np.average(a["E"][m & a["hit"]],
                                                weights=a["w"][m & a["hit"]]))
                               if (m & a["hit"]).any() else float("nan")),
        }
    w, th, hit = a["w"], a["theta_nd"], a["hit"]
    rows["ALL"] = {
        "pi0_per_pot": float(a["pi0_w"].sum() / a["pot"]),
        "median_dist_m": float(np.median(a["pi0_dist"])),
        "median_subtend_mrad": float(1e3 * np.median(a["pi0_subtend"])),
        "median_theta_nd_mrad": float(1e3 * np.median(th)),
        "frac_theta_lt_10mrad": float(w[th < 0.010].sum() / w.sum()),
        "acceptance": float(w[hit].sum() / w.sum()),
        "chi_to_nd_per_pot": float(2 * w[hit].sum() / a["pot"]),
        "mean_E_hit_GeV": float(np.average(a["E"][hit], weights=w[hit])),
    }
    return rows


def figure(a, run):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                          left=0.115, right=0.98, top=0.83, bottom=0.075)
    tgt = a["pi0_reg"] == 0
    s = summarise(a)

    # (a) theta to the ND, per region, shape-normalised
    ax = fig.add_subplot(gs[0, 0])
    b = np.geomspace(1e-4, 3.0, 90)
    for i, (name, c) in enumerate(REGIONS):
        m = a["reg"] == i
        if not m.any():
            continue
        h, _ = np.histogram(a["theta_nd"][m], bins=b, weights=a["w"][m])
        ax.step(0.5 * (b[1:] + b[:-1]), h / h.sum() / np.diff(b), where="mid",
                color=c, lw=1.5, label=f"{name}")
    ax.axvline(np.median(a["pi0_subtend"][tgt]), color="tab:orange", ls=":", lw=1.2)
    ax.axvline(np.median(a["pi0_subtend"][a["pi0_reg"] == 4]), color="tab:purple",
               ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_{\rm ND}$ = angle($p_\chi$, line to ND centre)  (rad)")
    ax.set_ylabel("normalised density")
    ax.set_title(r"(a) how far off the ND each $\chi$ points"
                 "\n(dotted: ND half-angle from target / from absorber)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) same, but scaled by the ND half-angle seen from that vertex
    ax = fig.add_subplot(gs[0, 1])
    b = np.geomspace(1e-2, 3e3, 90)
    ratio = a["theta_nd"] / a["subtend"]
    for i, (name, c) in enumerate(REGIONS):
        m = a["reg"] == i
        if not m.any():
            continue
        h, _ = np.histogram(ratio[m], bins=b, weights=a["w"][m])
        ax.step(0.5 * (b[1:] + b[:-1]), h / h.sum() / np.diff(b), where="mid",
                color=c, lw=1.5, label=name)
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_{\rm ND}$ / (ND half-angle from that vertex)")
    ax.set_ylabel("normalised density")
    ax.set_title("(b) the same angle measured in units of the detector's own\n"
                 "angular size — left of the dashed line is on target", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (c) acceptance and flux per region
    ax = fig.add_subplot(gs[1, 0])
    names = [n for n, _ in REGIONS if n in s]
    acc = np.array([s[n]["acceptance"] for n in names])
    flux = np.array([s[n]["chi_to_nd_per_pot"] for n in names])
    npi0 = np.array([s[n]["pi0_per_pot"] for n in names])
    y = np.arange(len(names))
    ax.barh(y, flux, 0.6,
            color=["tab:orange" if n == "target graphite" else "0.65" for n in names])
    for i, n in enumerate(names):
        ax.text(flux[i] * 1.15, i,
                f"{flux[i]:.2e}   =  {npi0[i]:5.2f} $\\pi^0$/POT  $\\times$ "
                f"{acc[i]:.1e} acc. $\\times$ 2",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 3.0)
    ax.set_xlabel(r"$\chi$ (+$\bar\chi$) crossing the ND per POT")
    ax.set_title(r"(c) what actually reaches the detector: $\pi^0$ rate $\times$"
                 " geometric acceptance", fontsize=10)
    ax.grid(alpha=0.25, axis="x")

    # (d) angle to the beam axis -- the intrinsic broadening, no geometry in it
    ax = fig.add_subplot(gs[1, 1])
    b = np.geomspace(1e-4, 3.0, 90)
    for i, (name, c) in enumerate(REGIONS):
        m = a["reg"] == i
        if not m.any():
            continue
        h, _ = np.histogram(a["theta_beam"][m], bins=b, weights=a["w"][m])
        ax.step(0.5 * (b[1:] + b[:-1]), h / h.sum() / np.diff(b), where="mid",
                color=c, lw=1.5, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ to the beam axis (rad)")
    ax.set_ylabel("normalised density")
    ax.set_title(r"(d) intrinsic $\chi$ angle to the beam axis", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    tg, al = s["target graphite"], s["ALL"]
    txt = (f"$m_\\chi$ = {1e3*a['m_chi']:.0f} MeV, $m_{{A'}}$ = 3$m_\\chi$ = "
           f"{1e3*a['m_ap']:.0f} MeV,  {a['n_sample']} decays sampled per $\\pi^0$\n"
           f"target graphite: {tg['pi0_per_pot']:.2f} $\\pi^0$/POT, median "
           f"$\\theta_{{\\rm ND}}$ = {tg['median_theta_nd_mrad']:.0f} mrad, "
           f"acceptance {tg['acceptance']:.2e}, "
           f"{tg['chi_to_nd_per_pot']:.2e} $\\chi$/POT at the ND\n"
           f"whole beamline: {al['pi0_per_pot']:.2f} $\\pi^0$/POT, median "
           f"$\\theta_{{\\rm ND}}$ = {al['median_theta_nd_mrad']:.0f} mrad "
           f"({al['median_theta_nd_mrad']/tg['median_theta_nd_mrad']:.1f}$\\times$ wider), "
           f"acceptance {al['acceptance']:.2e}, "
           f"{al['chi_to_nd_per_pot']:.2e} $\\chi$/POT "
           f"({al['chi_to_nd_per_pot']/tg['chi_to_nd_per_pot']:.1f}$\\times$ the target alone)")
    fig.text(0.5, 0.955, txt, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle(r"(7)  Angular spread of $\chi$ from $\pi^0\to\gamma A'\to\gamma\chi\bar\chi$"
                 " along the whole NuMI beamline"
                 f"   —   g4numi {run}", fontsize=12.5, y=0.998)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig7_dm_angular_spread_{run}_mchi{int(1e3*a['m_chi']):03d}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r0003")
    ap.add_argument("--mchi", type=float, default=0.012, help="GeV")
    ap.add_argument("--nsample", type=int, default=4)
    args = ap.parse_args()

    a = analyse(args.run, args.mchi, args.nsample)
    s = summarise(a)
    tag = f"{args.run}_mchi{int(1e3*args.mchi):03d}"
    (HERE / f"dm_angular_summary_{tag}.json").write_text(json.dumps(s, indent=2))
    figure(a, args.run)

    cols = ("pi0/POT", "dist m", "subtend mr", "med th mr", "f<10mr",
            "accept", "chi/POT", "<E> GeV")
    print(f"\n{'region':24s}" + "".join(f"{c:>12s}" for c in cols))
    for k, r in s.items():
        print(f"{k:24s}" + "".join(f"{r[v]:>12.4g}" for v in
                                   ("pi0_per_pot", "median_dist_m",
                                    "median_subtend_mrad", "median_theta_nd_mrad",
                                    "frac_theta_lt_10mrad", "acceptance",
                                    "chi_to_nd_per_pot", "mean_E_hit_GeV")))


if __name__ == "__main__":
    main()
