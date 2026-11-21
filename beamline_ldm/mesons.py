"""Beamline-simulation outputs -> inputs of a light-dark-matter generator (BdNMC `particle_list` format).

Subcommands (`--beamline g4numi` is the default; `g4lbnf` is the LBNF simulation):

    python -m beamline_ldm.mesons macro   --template baseline.mac --run-id 3 --pot 1000000 --out run.mac
    python -m beamline_ldm.mesons mesons  g4numi_nudata_0003.root --pot 1000000 --run-id 3 --out-dir particle_lists
    python -m beamline_ldm.mesons mesons  lbnf_00011.mesons.root --beamline g4lbnf --run-id 11 --split-regions
    python -m beamline_ldm.mesons regions lbnf_00011.mesons.root --beamline g4lbnf
    python -m beamline_ldm.mesons protons zpNtuple_0003.root --pot 1000000 [--interacting]     # g4numi only

`mesons` reads the `meson` tree (px py pz E in GeV, x y z in cm, t in ns, weight, pdg), unweights each species by
resampling with replacement (seed 42 + run id) and writes one list per species: `px py pz E` (GeV), `x y z` (m, z
relative to the target centre) and `t` (s), with `yield_per_POT=` in the header. `--region R` keeps only mesons made
in region R (file suffix `_R`); `--split-regions` writes every region's lists plus the unsplit ones. Region yields
come from the unfiltered pool, so the region lists' yields add up to the total.
`regions` prints the weighted yield per POT of each species in each source region.
`protons` reads g4numi's `zp` tree (z-plane crossings) and returns the yield per POT of protons with pz > 100 GeV/c
crossing the absorber planes, split into the components a BdNMC bremsstrahlung card is written for.

When the input has the beamline's metadata tree (`zpmeta` for g4numi, `meta` for g4lbnf) its `pots` value
replaces --pot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import beamlines
from .beamlines import G4NUMI_ABSORBER_PLANES_CM as ABSORBER_PLANES_CM
from .beamlines import G4NUMI_TARGET_CENTER_Z as TARGET_CENTER_Z_G4NUMI

# g4numi absorber core window (g4numi coordinates, cm), from the g4numi geometry.
CORE_HALF_X_CM = 51.0 * 2.54 / 2.0
CORE_HALF_Y_CM = 51.0 * 2.54 / (2.0 * math.cos(0.05835))  # tilt projected
CORE_Y_CENTRE_CM = -10.8

SPECIES = {111: "pi0", 221: "eta", 113: "rho0", 223: "omega", 333: "phi", 331: "etaprime"}

G4NUMI_MACRO_OVERRIDES = ("/NuMI/output/outputNuNtuple", "/NuMI/output/setNuNtupleFile",
                          "/NuMI/output/outputZpNtuple", "/NuMI/output/setZpNtupleFile", "/NuMI/rndm/setRndmSeed",
                          "/NuMI/run/setRunID", "/run/beamOn")
MACRO_OVERRIDES = G4NUMI_MACRO_OVERRIDES  # backward-compatible name
G4LBNF_MACRO_OVERRIDES = ("/LBNE/run/NEvents", "/LBNE/run/setRunID", "/LBNE/rndm/setRndmSeed",
                          "/LBNE/mesons/", "/run/beamOn", "exit")


def write_macro(template: Path, out: Path, run_id: int, pot: int, beamline: str = "g4numi",
                stem: str = "lbnf") -> Path:
    """Copy a run macro, replacing the output, seed, run-id and event-count lines.
    Output file names stay short and relative (Geant4 truncates long ones)."""
    text = template.read_text().splitlines(keepends=True)
    if beamline == "g4numi":
        lines = [ln for ln in text if not any(k in ln for k in G4NUMI_MACRO_OVERRIDES)]
        lines += ["\n# LDM run settings\n",
                  "/NuMI/output/outputNuNtuple true\n", "/NuMI/output/setNuNtupleFile g4numi_nudata\n",
                  "/NuMI/output/outputZpNtuple true\n", "/NuMI/output/setZpNtupleFile zpNtuple\n",
                  f"/NuMI/rndm/setRndmSeed {run_id}\n", f"/NuMI/run/setRunID {run_id}\n", f"/run/beamOn {pot}\n"]
    elif beamline == "g4lbnf":
        lines = [ln for ln in text if not ln.lstrip().startswith(G4LBNF_MACRO_OVERRIDES)]
        lines += ["\n# LDM run settings (g4lbnf: beamOn without an argument runs NEvents events)\n",
                  f"/LBNE/run/setRunID {run_id}\n", f"/LBNE/rndm/setRndmSeed {run_id}\n",
                  f"/LBNE/run/NEvents {pot}\n", f"/LBNE/mesons/FileName {stem}\n", "/LBNE/mesons/Create True\n",
                  "/run/beamOn\n", "exit\n"]
    else:
        raise KeyError(f"no macro writer for beamline {beamline!r}")
    out.write_text("".join(lines))
    return out


def simulated_pot(root_file: Path, default: float, beamline: str = "g4numi") -> float:
    import uproot
    tree = beamlines.get(beamline).pot_tree
    with uproot.open(root_file) as f:
        for name in (tree, "zpmeta", "meta"):
            if name in f and "pots" in f[name]:
                return float(f[name]["pots"].array(library="np")[0])
    return float(default)


def read_names(f) -> Optional[dict]:
    """{(kind, id): name} from a MesonTally `names` tree, or None."""
    if "names" not in f:
        return None
    t = f["names"]
    ids = t["id"].array(library="np")
    kinds = t["kind"].array(library="np")
    names = t["name"].array(library="np")
    return {(int(k), int(i)): str(n) for k, i, n in zip(kinds, ids, names)}


def read_mesons(meson_file: Path):
    import uproot
    with uproot.open(meson_file) as f:
        if "meson" not in f:
            raise KeyError(f"'meson' tree not found in {meson_file} (enable the meson output of the LDM patches)")
        t = f["meson"]
        keys = [k for k in ("px", "py", "pz", "E", "x", "y", "z", "t", "weight", "pdg", "vol", "mat") if k in t]
        a = {k: t[k].array(library="np") for k in keys}
        names = read_names(f)
    return a, names


def region_yields(meson_file: Path, pots: float, beamline: str = "g4numi") -> dict:
    """{species: {region: weighted yield per POT}} including a "total" entry per species."""
    a, names = read_mesons(meson_file)
    reg = beamlines.get(beamline).regions(a, names)
    out = {}
    for pdg, name in SPECIES.items():
        m = a["pdg"] == pdg
        if not m.any():
            continue
        d = {"total": float(a["weight"][m].sum() / pots)}
        for r in sorted(set(reg[m])):
            d[str(r)] = float(a["weight"][m & (reg == r)].sum() / pots)
        out[name] = d
    return out


def extract_mesons(meson_file: Path, out_dir: Path, run_id: int, pots: float, beamline: str = "g4numi",
                   region: Optional[str] = None, target_center_z_cm: Optional[float] = None) -> dict:
    """Unweighted particle lists per species; returns the weighted yield per POT of each.
    With `region`, keep only mesons made in that source region (yields are then that region's share)."""
    bl = beamlines.get(beamline)
    z0 = bl.target_center_z_cm if target_center_z_cm is None else target_center_z_cm
    out_dir.mkdir(parents=True, exist_ok=True)
    a, names = read_mesons(meson_file)
    sel = np.ones(len(a["pdg"]), dtype=bool)
    suffix = ""
    if region is not None:
        sel = bl.regions(a, names) == region
        suffix = f"_{region}"
    yields = {}
    for pdg, name in SPECIES.items():
        mask = (a["pdg"] == pdg) & sel
        n_raw = int(mask.sum())
        out_file = out_dir / f"{name}_list_{run_id:04d}{suffix}.dat"
        if n_raw == 0:
            yields[name] = 0.0
            out_file.write_text(f"# px py pz E (GeV) x y z (meters) t (seconds) - dummy placeholder for empty {name}\n"
                                "# yield_per_POT=0.0000e00\n" + "0.0e00 " * 7 + "0.0e00\n")
            continue
        w = a["weight"][mask]
        yields[name] = float(w.sum() / pots)
        np.random.seed(42 + run_id)  # legacy global seeding: reproduces the production lists bit for bit
        idx = np.random.choice(n_raw, size=n_raw, replace=True, p=w / w.sum())
        cols = np.column_stack([a["px"][mask][idx], a["py"][mask][idx], a["pz"][mask][idx], a["E"][mask][idx],
                                a["x"][mask][idx] / 100.0, a["y"][mask][idx] / 100.0,
                                (a["z"][mask][idx] - z0) / 100.0, a["t"][mask][idx] / 1e9])
        header = ("px py pz E (GeV) x y z (meters) t (seconds) - extracted and unweighted\n"
                  f"yield_per_POT={yields[name]:.6e}")
        np.savetxt(out_file, cols, fmt="%.6e", header=header)
    return yields


def proton_yields(zp_file: Path, pots: float, planes: dict = ABSORBER_PLANES_CM,
                  core=(CORE_HALF_X_CM, CORE_HALF_Y_CM, CORE_Y_CENTRE_CM), interacting: bool = False) -> dict:
    """g4numi: yield per POT of protons (pz > 100 GeV/c) at the absorber planes, per brem component.
    interacting=True converts plane crossings into protons that interact in each layer (no double counting)."""
    import uproot
    with uproot.open(zp_file) as f:
        zp = f["zp"]
        zpoint = zp["zpdata/zpoint"].array(library="np")
        pid = zp["zpdata/pidtype"].array(library="np")
        x = zp["zpdata/xposatz"].array(library="np")
        y = zp["zpdata/yposatz"].array(library="np")
        pz = zp["zpdata/zmomatz"].array(library="np")
        w = zp["zpdata/nimpwt"].array(library="np") if "zpdata/nimpwt" in zp else np.ones(len(pid))
    hx, hy, yc = core
    inside = (np.abs(x) <= hx) & (np.abs(y - yc) <= hy)
    protons = (pid == 14) & (pz > 100.0)

    def at(z, region=None):
        m = protons & (np.abs(zpoint - z) < 1.0)
        if region == "inside":
            m &= inside
        elif region == "outside":
            m &= ~inside
        return float(w[m].sum() / pots)

    y = {"concrete": at(planes["concrete"]), "al_core": at(planes["core_front"], "inside"),
         "blublocks": at(planes["core_front"], "outside"), "steel_core": at(planes["core_mid"], "inside")}
    if interacting:
        y = {"concrete": max(0.0, y["concrete"] - (y["al_core"] + y["blublocks"])),
             "al_core": max(0.0, y["al_core"] - y["steel_core"]),
             "blublocks": y["blublocks"], "steel_core": y["steel_core"]}
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def beamline_arg(p):
        p.add_argument("--beamline", default="g4numi", choices=sorted(beamlines.BEAMLINES))

    m = sub.add_parser("macro", help="run macro for one run")
    beamline_arg(m)
    m.add_argument("--template", type=Path, required=True)
    m.add_argument("--run-id", type=int, required=True)
    m.add_argument("--pot", type=int, required=True)
    m.add_argument("--stem", default="lbnf", help="g4lbnf: meson file stem (<stem>_<run>.mesons.root)")
    m.add_argument("--out", type=Path, default=Path("run.mac"))
    s = sub.add_parser("mesons", help="unweighted meson lists from the meson tree")
    beamline_arg(s)
    s.add_argument("root", type=Path)
    s.add_argument("--pot", type=float, default=0.0, help="simulated POT (replaced by the metadata tree if present)")
    s.add_argument("--run-id", type=int, required=True)
    s.add_argument("--out-dir", type=Path, default=Path("particle_lists"))
    s.add_argument("--region", default=None, help="keep only mesons made in this source region")
    s.add_argument("--split-regions", action="store_true", help="also write one set of lists per source region")
    s.add_argument("--target-z", type=float, default=None, help="target centre z in cm (default: the beamline's)")
    r = sub.add_parser("regions", help="weighted yield per POT by species and source region")
    beamline_arg(r)
    r.add_argument("root", type=Path)
    r.add_argument("--pot", type=float, default=0.0)
    p = sub.add_parser("protons", help="g4numi: proton yields at the absorber planes")
    p.add_argument("root", type=Path)
    p.add_argument("--pot", type=float, required=True)
    p.add_argument("--interacting", action="store_true")
    p.add_argument("--planes", type=json.loads, default=None, help='JSON {"concrete": z, "core_front": z, "core_mid": z} (cm)')
    p.add_argument("--core", type=float, nargs=3, default=None, metavar=("HALF_X", "HALF_Y", "Y_CENTRE"))
    args = ap.parse_args(argv)

    if args.cmd == "macro":
        print(write_macro(args.template, args.out, args.run_id, args.pot, args.beamline, args.stem))
    elif args.cmd in ("mesons", "regions"):
        pots = simulated_pot(args.root, args.pot, args.beamline)
        if pots <= 0:
            ap.error("no POT: the file has no metadata tree; pass --pot")
        if args.cmd == "regions":
            print(json.dumps(region_yields(args.root, pots, args.beamline), indent=2))
            return 0
        res = {"all": extract_mesons(args.root, args.out_dir, args.run_id, pots, args.beamline, args.region,
                                     args.target_z)}
        if args.split_regions and args.region is None:
            a, names = read_mesons(args.root)
            for reg in sorted(set(beamlines.get(args.beamline).regions(a, names))):
                res[reg] = extract_mesons(args.root, args.out_dir, args.run_id, pots, args.beamline, reg,
                                          args.target_z)
        print(json.dumps(res if args.split_regions else res["all"], indent=2))
    else:
        kw = {}
        if args.planes:
            kw["planes"] = args.planes
        if args.core:
            kw["core"] = tuple(args.core)
        print(json.dumps(proton_yields(args.root, simulated_pot(args.root, args.pot), interacting=args.interacting,
                                       **kw), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
