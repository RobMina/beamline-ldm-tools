"""Beamline backends: what differs between Geant4 beamline simulations when turning their meson output into
light-dark-matter generator inputs.

Every backend reads the same `meson` tree (px py pz E in GeV, x y z in cm, t in ns, weight, pdg; the schema of
the g4numi LDM patches and of `geant4/MesonTally`) and differs only in

- the file's POT source (a metadata tree),
- the target centre z (the origin of the target-centred list frame),
- how a production vertex maps to a source region (target, horns, decay pipe, absorber, ...),
- the run-macro commands.

Add a beamline by registering another `Beamline` in `BEAMLINES`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

# --------------------------------------------------------------------------------------------------- g4numi (NuMI)
# g4numi target centre z (cm): the ME target spans -143.3 to -20.0 cm.
G4NUMI_TARGET_CENTER_Z = -81.65
# Hadron-absorber planes (g4numi coordinates, cm), from the g4numi geometry.
G4NUMI_ABSORBER_PLANES_CM = {"concrete": 72183.4,   # concrete pre-absorber front face
                             "core_front": 72366.0,  # absorber front face (aluminium / outer-block boundary)
                             "core_mid": 72641.7}    # aluminium / steel core boundary


def _g4numi_regions(a: dict, names: Optional[dict]) -> np.ndarray:
    """g4numi's meson tree has no volume branch by default: split by z at the pre-absorber concrete face.
    "upstream" is target + horns + decay pipe."""
    return np.where(a["z"] >= G4NUMI_ABSORBER_PLANES_CM["concrete"], "absorber", "upstream")


# --------------------------------------------------------------------------------------------------- g4lbnf (LBNF)
# OfficialEngDesignSept2021 (1.5 m RAL target): the graphite core spans z = 0-150 cm in g4lbnf global coordinates.
G4LBNF_TARGET_CENTER_Z = 75.0
# Downstream of this z (cm) and outside the decay pipe is the hadron absorber (gdml/TOP.gdml, entries at 225-230 m).
G4LBNF_ABSORBER_Z_CM = 21800.0


def _g4lbnf_regions(a: dict, names: Optional[dict]) -> np.ndarray:
    """Region from the creation volume / material written by MesonTally (`vol`, `mat` ids into the `names` tree):
    target = graphite ("Target" material); decaypipe = *DecayPipe* volumes; rock = RockLogical; absorber = anything
    else beyond G4LBNF_ABSORBER_Z_CM; horns = everything else upstream (horns, baffle, target hall, windows)."""
    n = len(a["z"])
    if names is None or "vol" not in a or "mat" not in a:
        return np.where(a["z"] >= G4LBNF_ABSORBER_Z_CM, "absorber", "upstream")
    vol = np.array([names.get((1, int(i)), "") for i in a["vol"]], dtype=object)
    mat = np.array([names.get((2, int(i)), "") for i in a["mat"]], dtype=object)
    out = np.full(n, "horns", dtype=object)
    out[a["z"] >= G4LBNF_ABSORBER_Z_CM] = "absorber"
    out[vol == "RockLogical"] = "rock"
    out[np.array(["DecayPipe" in v for v in vol], dtype=bool)] = "decaypipe"
    out[mat == "Target"] = "target"
    return out.astype(str)


@dataclass(frozen=True)
class Beamline:
    name: str
    target_center_z_cm: float
    pot_tree: str            # metadata tree holding `pots`
    regions: Callable[[dict, Optional[dict]], np.ndarray]
    macro_prefix: str        # Geant4 UI directory of the application
    notes: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


BEAMLINES: Dict[str, Beamline] = {
    "g4numi": Beamline("g4numi", G4NUMI_TARGET_CENTER_Z, "zpmeta", _g4numi_regions, "/NuMI",
                       notes="NuMI; meson tree from the RobMina/g4numi `ldm` branch (g4numi_nudata_NNNN.root)",
                       extra=G4NUMI_ABSORBER_PLANES_CM),
    "g4lbnf": Beamline("g4lbnf", G4LBNF_TARGET_CENTER_Z, "meta", _g4lbnf_regions, "/LBNE",
                       notes="LBNF; meson tree from geant4/MesonTally in g4lbne (<stem>_NNNNN.mesons.root)",
                       extra={"absorber_z": G4LBNF_ABSORBER_Z_CM}),
}


def get(name: str) -> Beamline:
    try:
        return BEAMLINES[name]
    except KeyError:
        raise KeyError(f"unknown beamline {name!r}; known: {', '.join(BEAMLINES)}") from None
