#!/usr/bin/env python
"""Where do the pions g4numi makes outside the NuMI target come from?

Needs a zpNtuple written by a g4numi run with
`/NuMI/output/outputMesonProvenance true`, which adds ancestry branches to the
`meson` tree (see NumiAnalysis::RecordTrack / FillMesonProvenance):

    proc, vol, intgt      creator process, production volume, in-target flag
    ppdg, pproc, pv{x,y,z}, pE          the immediate parent
    cflag, cpdg, cproc, cE, cv{x,y,z}   the "carrier"
    iv{x,y,z}                           where the chain left the target
    (provnames tree: id <-> name legend for proc / vol)

The carrier is the last ancestor born inside the target whose next-in-chain
daughter was born outside it -- the particle that actually took the energy out
of the target.  cflag says which of two stories applies:

    cflag = 1   the beam proton interacted in the target, and one of the
                secondaries it made there escaped and started a shower
                somewhere else -- "secondary radiation out of the target"
    cflag = 2   nothing in this pion's ancestry was born in the target at all,
                so the beam proton itself carried 120 GeV past the target and
                first interacted downstream -- "the proton punched through"

Usage:
    python pion_provenance.py [--file .../zpNtuple_0030.root]
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

PDG_LABEL = {2212: "p", 2112: "n", 211: r"$\pi^+$", -211: r"$\pi^-$",
             111: r"$\pi^0$", 321: r"$K^+$", -321: r"$K^-$", 130: r"$K^0_L$",
             310: r"$K^0_S$", 22: r"$\gamma$", 3122: r"$\Lambda$",
             -2212: r"$\bar p$", -2112: r"$\bar n$", 3222: r"$\Sigma^+$",
             3112: r"$\Sigma^-$", 3212: r"$\Sigma^0$", -3122: r"$\bar\Lambda$",
             13: r"$\mu^-$", -13: r"$\mu^+$", 221: r"$\eta$"}

# z landmarks, cm -- same partition mipp_common uses for the beamline profile
ZONES = [("upstream / baffle", -1e9, -143.3),
         ("target", -143.3, -20.0),
         ("horn 1", -20.0, 400.0),
         ("horn 2 + hall", 400.0, 4500.0),
         ("decay pipe", 4500.0, 67500.0),
         ("shielding", 67500.0, 72000.0),
         ("absorber", 72000.0, 1e9)]


def load(path):
    import uproot
    f = uproot.open(path)
    pot = float(f["zpmeta"].arrays(library="np")["pots"][0])
    t = f[sorted(k for k in f.keys() if k.startswith("meson;"))[-1]]
    a = t.arrays(library="np")
    leg = f["provnames"].arrays(library="np")
    names = {}
    for i, k, n in zip(leg["id"], leg["kind"], leg["name"]):
        names[(int(k), int(i))] = n.decode() if isinstance(n, bytes) else str(n)
    return pot, a, names


def wtop(vals, w, labeller, n=10):
    """Weight-ordered breakdown of a categorical column."""
    ids, inv = np.unique(vals, return_inverse=True)
    s = np.bincount(inv, weights=w)
    order = np.argsort(-s)
    tot = s.sum()
    rows = [(labeller(int(ids[j])), float(s[j]), float(s[j] / tot)) for j in order[:n]]
    other = float(tot - sum(r[1] for r in rows))
    if other > 0:
        rows.append(("other", other, other / tot))
    return rows, float(tot)


def zone_of(z):
    r = np.full(len(z), -1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(ZONES):
        r[(z >= lo) & (z < hi)] = i
    return r


def analyse(path, species):
    pot, a, names = load(path)
    pname = lambda i: names.get((0, i), f"proc{i}")
    vname = lambda i: names.get((1, i), f"vol{i}")
    lname = lambda i: PDG_LABEL.get(i, str(i))

    sel = np.isin(a["pdg"], species)
    w = a["weight"][sel]
    out = a["intgt"][sel] == 0
    wo = w[out]
    res = {
        "pot": pot,
        "pions_per_pot": float(w.sum() / pot),
        "in_target_per_pot": float(w[~out].sum() / pot),
        "outside_per_pot": float(wo.sum() / pot),
    }

    res["creator_process"], _ = wtop(a["proc"][sel][out], wo, pname, 12)
    res["production_volume"], _ = wtop(a["vol"][sel][out], wo, vname, 12)

    cflag = a["cflag"][sel][out]
    res["origin_story"] = {
        "beam proton interacted in the target; a secondary escaped (cflag=1)":
            float(wo[cflag == 1].sum() / wo.sum()),
        "no ancestor born in the target; the proton itself went past (cflag=2)":
            float(wo[cflag == 2].sum() / wo.sum()),
        "unresolved chain (cflag=-1)": float(wo[cflag == -1].sum() / wo.sum()),
    }

    # what the escaping particle was, and where it then interacted
    esc = out & (a["cflag"][sel] == 1)
    res["target_escape_carrier"], _ = wtop(a["cpdg"][sel][esc], w[esc], lname, 10)
    res["carrier_interaction_zone"], _ = wtop(
        zone_of(a["ivz"][sel][esc]), w[esc], lambda i: ZONES[i][0] if i >= 0 else "?", 8)
    ce = a["cE"][sel][esc]
    res["carrier_energy_GeV"] = {
        "mean": float(np.average(ce, weights=w[esc])),
        "median": float(np.interp(0.5 * w[esc].sum(),
                                  np.cumsum(w[esc][np.argsort(ce)]), np.sort(ce))),
        "frac_above_5": float(w[esc][ce > 5].sum() / w[esc].sum()),
    }

    punch = out & (a["cflag"][sel] == 2)
    res["punchthrough_first_interaction_zone"], _ = wtop(
        zone_of(a["ivz"][sel][punch]), w[punch],
        lambda i: ZONES[i][0] if i >= 0 else "?", 8)

    # immediate parent of the pion, and the parent's own birthplace
    res["immediate_parent"], _ = wtop(a["ppdg"][sel][out], wo, lname, 10)
    res["parent_born_in_target_frac"] = float(
        wo[a["pintgt"][sel][out] == 1].sum() / wo.sum())

    # forward pions only -- the ones that could matter for the DM signal
    fwd = out & (a["pz"][sel] > 4.0)
    if fwd.any():
        res["forward_pz_gt_4"] = {
            "per_pot": float(w[fwd].sum() / pot),
            "creator_process": wtop(a["proc"][sel][fwd], w[fwd], pname, 8)[0],
            "production_volume": wtop(a["vol"][sel][fwd], w[fwd], vname, 8)[0],
            "origin_story_cflag2_frac":
                float(w[fwd][a["cflag"][sel][fwd] == 2].sum() / w[fwd].sum()),
            "carrier": wtop(a["cpdg"][sel][fwd], w[fwd], lname, 8)[0],
        }
    return res, (pot, a, names, sel, out)


def figure(res, blob, tag):
    pot, a, names, sel, out = blob
    pname = lambda i: names.get((0, i), f"proc{i}")
    vname = lambda i: names.get((1, i), f"vol{i}")
    lname = lambda i: PDG_LABEL.get(i, str(i))
    w = a["weight"][sel]
    wo = w[out]

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42,
                          left=0.13, right=0.98, top=0.83, bottom=0.07)

    def barh(ax, rows, title, color="0.6", nmax=9):
        rows = rows[:nmax][::-1]
        y = np.arange(len(rows))
        v = np.array([r[2] for r in rows])
        ax.barh(y, v, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        for i, r in enumerate(rows):
            ax.text(v[i] + 0.008, i, f"{100*r[2]:.1f}%", va="center", fontsize=7.5)
        ax.set_xlim(0, min(1.0, v.max() * 1.32))
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="x")

    barh(fig.add_subplot(gs[0, 0]), res["creator_process"],
         "(a) creator process of the pion", "tab:blue")
    barh(fig.add_subplot(gs[0, 1]), res["production_volume"],
         "(b) volume it was made in", "tab:green")
    barh(fig.add_subplot(gs[0, 2]), res["target_escape_carrier"],
         "(c) what carried the energy out of the target\n(cflag = 1 only)",
         "tab:orange")
    barh(fig.add_subplot(gs[1, 0]), res["immediate_parent"],
         "(d) immediate parent of the pion", "tab:red")
    barh(fig.add_subplot(gs[1, 1]), res["carrier_interaction_zone"],
         "(e) where that carrier then interacted", "tab:purple")

    # (f) carrier energy spectrum, split by story
    ax = fig.add_subplot(gs[1, 2])
    b = np.geomspace(0.05, 200, 60)
    for cf, lab, c in ((1, "secondary out of the target", "tab:orange"),
                       (2, "beam proton past the target", "tab:cyan")):
        m = out & (a["cflag"][sel] == cf)
        if not m.any():
            continue
        h, _ = np.histogram(a["cE"][sel][m], bins=b, weights=w[m])
        ax.step(0.5 * (b[1:] + b[:-1]), h / pot / np.diff(b), where="mid",
                color=c, lw=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("carrier energy at its birth (GeV)")
    ax.set_ylabel("pions / POT per GeV")
    ax.set_title("(f) energy of the particle that left the target", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    st = res["origin_story"]
    k1 = [v for k, v in st.items() if "cflag=1" in k][0]
    k2 = [v for k, v in st.items() if "cflag=2" in k][0]
    ce = res["carrier_energy_GeV"]
    txt = (f"{res['outside_per_pot']:.1f} pions/POT are made outside the target "
           f"({100*res['outside_per_pot']/res['pions_per_pot']:.0f} % of "
           f"{res['pions_per_pot']:.1f}/POT over the whole beamline)\n"
           f"{100*k1:.0f} % descend from a secondary that was born in the target "
           f"and escaped it "
           f"(median energy {ce['median']:.1f} GeV, {100*ce['frac_above_5']:.0f} % above 5 GeV)\n"
           f"{100*k2:.0f} % have no target-born ancestor at all: the beam proton "
           f"itself crossed the target without seeding them and interacted downstream")
    fig.text(0.5, 0.955, txt, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    fig.suptitle("(8)  Provenance of the pions g4numi makes outside the NuMI target"
                 f"   —   {tag}", fontsize=12.5, y=0.998)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"fig8_pion_provenance_{tag}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tag", default="prov_me")
    ap.add_argument("--species", default="pi", choices=("pi", "pi0", "pipm"))
    args = ap.parse_args()
    species = {"pi": (211, -211, 111), "pi0": (111,), "pipm": (211, -211)}[args.species]

    res, blob = analyse(Path(args.file).resolve(), species)
    suffix = args.tag if args.species == "pi" else f"{args.tag}_{args.species}"
    (HERE / f"pion_provenance_{suffix}.json").write_text(json.dumps(res, indent=2))
    figure(res, blob, suffix)

    print(f"\npions ({args.species}) per POT: {res['pions_per_pot']:.2f}"
          f"   in target {res['in_target_per_pot']:.2f}"
          f"   outside {res['outside_per_pot']:.2f}")
    for k, v in res["origin_story"].items():
        print(f"  {100*v:5.1f} %  {k}")
    for key in ("creator_process", "production_volume", "target_escape_carrier",
                "carrier_interaction_zone", "immediate_parent",
                "punchthrough_first_interaction_zone"):
        print(f"\n{key}")
        for name, val, frac in res[key]:
            print(f"   {name:34s} {val/res['pot']:8.3f}/POT {100*frac:6.1f}%")
    print("\ncarrier energy (GeV):", res["carrier_energy_GeV"])
    if "forward_pz_gt_4" in res:
        f = res["forward_pz_gt_4"]
        print(f"\nforward (pz > 4 GeV/c) pions outside the target: {f['per_pot']:.3f}/POT, "
              f"{100*f['origin_story_cflag2_frac']:.0f} % from a proton that "
              f"crossed the target")
        for key in ("creator_process", "production_volume", "carrier"):
            print(f"  {key}: " + ", ".join(f"{n} {100*fr:.0f}%" for n, _, fr in f[key][:5]))


if __name__ == "__main__":
    main()
