"""Backward-compatible entry point: `python -m g4numi_ldm.mesons ...` = `python -m beamline_ldm.mesons ...`
with the g4numi backend (the default). New code: `beamline_ldm.mesons`."""
import sys

from beamline_ldm.mesons import *  # noqa: F401,F403
from beamline_ldm.mesons import (ABSORBER_PLANES_CM, MACRO_OVERRIDES, SPECIES, TARGET_CENTER_Z_G4NUMI,  # noqa: F401
                                 extract_mesons, main, proton_yields, simulated_pot, write_macro)

if __name__ == "__main__":
    sys.exit(main())
