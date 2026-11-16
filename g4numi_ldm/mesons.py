"""g4numi outputs -> inputs of a light-dark-matter generator (BdNMC `particle_list` format).

Three steps, each a subcommand:

    python -m g4numi_ldm.mesons macro   --template macros/baseline.mac --run-id 3 --pot 1000000 --out run.mac
    python -m g4numi_ldm.mesons mesons  g4numi_nudata_0003.root --pot 1000000 --run-id 3 --out-dir particle_lists
    python -m g4numi_ldm.mesons protons zpNtuple_0003.root --pot 1000000 [--interacting]

`mesons` reads the `meson` tree written by the LDM-patched g4numi (px py pz E in GeV, x y z in cm, t in ns, weight,
pdg), unweights each species by resampling with replacement (seed 42 + run id) and writes one list per species:
`px py pz E` (GeV), `x y z` (m, z relative to the target centre) and `t` (s), with `# yield_per_POT=` in the header.
`protons` reads the `zp` tree (z-plane crossings) and returns the yield per POT of protons with pz > 100 GeV/c
crossing the absorber planes, split into the components a BdNMC bremsstrahlung card is written for.

The z-planes and the core window default to the NuMI hadron-absorber geometry of g4numi's own geometry code
(g4numi coordinates, cm); pass --planes / --core to change them. When the input has a `zpmeta` tree its `pots`
value replaces --pot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# g4numi target centre z (cm): the ME target spans -143.3 to -20.0 cm.
TARGET_CENTER_Z_G4NUMI = -81.65

# Hadron-absorber planes and core window (g4numi coordinates, cm), from the g4numi geometry.
ABSORBER_PLANES_CM = {"concrete": 72183.4,   # concrete pre-absorber front face
                      "core_front": 72366.0,  # absorber front face (aluminium / outer-block boundary)
                      "core_mid": 72641.7}    # aluminium / steel core boundary
CORE_HALF_X_CM = 51.0 * 2.54 / 2.0
CORE_HALF_Y_CM = 51.0 * 2.54 / (2.0 * math.cos(0.05835))  # tilt projected
CORE_Y_CENTRE_CM = -10.8

SPECIES = {111: "pi0", 221: "eta", 113: "rho0", 223: "omega", 333: "phi", 331: "etaprime"}

MACRO_OVERRIDES = ("/NuMI/output/outputNuNtuple", "/NuMI/output/setNuNtupleFile", "/NuMI/output/outputZpNtuple",
                   "/NuMI/output/setZpNtupleFile", "/NuMI/rndm/setRndmSeed", "/NuMI/run/setRunID", "/run/beamOn")


def write_macro(template: Path, out: Path, run_id: int, pot: int) -> Path:
    """Copy a g4numi macro, replacing the output, seed, run-id and beamOn lines.
    Output file names stay short and relative (Geant4 truncates long ones)."""
    lines = [ln for ln in template.read_text().splitlines(keepends=True)
             if not any(k in ln for k in MACRO_OVERRIDES)]
    lines += ["\n# LDM run settings\n",
              "/NuMI/output/outputNuNtuple true\n", "/NuMI/output/setNuNtupleFile g4numi_nudata\n",
              "/NuMI/output/outputZpNtuple true\n", "/NuMI/output/setZpNtupleFile zpNtuple\n",
              f"/NuMI/rndm/setRndmSeed {run_id}\n", f"/NuMI/run/setRunID {run_id}\n", f"/run/beamOn {pot}\n"]
    out.write_text("".join(lines))
    return out


def simulated_pot(root_file: Path, default: float) -> float:
    import uproot
    with uproot.open(root_file) as f:
        if "zpmeta" in f:
            return float(f["zpmeta"]["pots"].array(library="np")[0])
    return float(default)


def extract_mesons(meson_file: Path, out_dir: Path, run_id: int, pots: float) -> dict:
    """Unweighted particle lists per species; returns the weighted yield per POT of each."""
    import uproot
    out_dir.mkdir(parents=True, exist_ok=True)
    with uproot.open(meson_file) as f:
        if "meson" not in f:
            raise KeyError(f"'meson' tree not found in {meson_file} (enable the meson output of the LDM patches)")
        t = f["meson"]
        a = {k: t[k].array(library="np") for k in ("px", "py", "pz", "E", "x", "y", "z", "t", "weight", "pdg")}
    yields = {}
    for pdg, name in SPECIES.items():
        mask = a["pdg"] == pdg
        n_raw = int(mask.sum())
        out_file = out_dir / f"{name}_list_{run_id:04d}.dat"
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
                                (a["z"][mask][idx] - TARGET_CENTER_Z_G4NUMI) / 100.0, a["t"][mask][idx] / 1e9])
        header = ("px py pz E (GeV) x y z (meters) t (seconds) - extracted and unweighted\n"
                  f"yield_per_POT={yields[name]:.6e}")
        np.savetxt(out_file, cols, fmt="%.6e", header=header)
    return yields


def proton_yields(zp_file: Path, pots: float, planes: dict = ABSORBER_PLANES_CM,
                  core=(CORE_HALF_X_CM, CORE_HALF_Y_CM, CORE_Y_CENTRE_CM), interacting: bool = False) -> dict:
    """Yield per POT of protons (pz > 100 GeV/c) at the absorber planes, per brem component.
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
    m = sub.add_parser("macro", help="g4numi macro for one run")
    m.add_argument("--template", type=Path, required=True)
    m.add_argument("--run-id", type=int, required=True)
    m.add_argument("--pot", type=int, required=True)
    m.add_argument("--out", type=Path, default=Path("run.mac"))
    s = sub.add_parser("mesons", help="unweighted meson lists from the meson tree")
    s.add_argument("root", type=Path)
    s.add_argument("--pot", type=float, required=True, help="simulated POT (replaced by zpmeta if present)")
    s.add_argument("--run-id", type=int, required=True)
    s.add_argument("--out-dir", type=Path, default=Path("particle_lists"))
    p = sub.add_parser("protons", help="proton yields at the absorber planes")
    p.add_argument("root", type=Path)
    p.add_argument("--pot", type=float, required=True)
    p.add_argument("--interacting", action="store_true")
    p.add_argument("--planes", type=json.loads, default=None, help='JSON {"concrete": z, "core_front": z, "core_mid": z} (cm)')
    p.add_argument("--core", type=float, nargs=3, default=None, metavar=("HALF_X", "HALF_Y", "Y_CENTRE"))
    args = ap.parse_args(argv)
    if args.cmd == "macro":
        print(write_macro(args.template, args.out, args.run_id, args.pot))
    elif args.cmd == "mesons":
        print(json.dumps(extract_mesons(args.root, args.out_dir, args.run_id, simulated_pot(args.root, args.pot)),
                         indent=2))
    else:
        kw = {}
        if args.planes:
            kw["planes"] = args.planes
        if args.core:
            kw["core"] = tuple(args.core)
        print(json.dumps(proton_yields(args.root, simulated_pot(args.root, args.pot), interacting=args.interacting,
                                       **kw), indent=2))


if __name__ == "__main__":
    sys.exit(main())
