# g4numi-ldm-tools

Light-dark-matter (LDM) tools around [g4numi](https://github.com/NuSoftHEP/g4numi), the Geant4 simulation of the
NuMI beamline: turning g4numi's meson and z-plane outputs into the inputs of an LDM generator such as
[BdNMC](https://github.com/RobMina/BdNMC), and checking g4numi's charged-pion yields off the NuMI target against the
MIPP measurement. Nothing here is specific to one detector; a downstream detector's placement is a configuration
input.

The meson, z-plane, target-exit and shower-seed outputs these tools read are added to g4numi on the `ldm` branch of
the fork [RobMina/g4numi](https://github.com/RobMina/g4numi/tree/ldm) (five commits on upstream `main_modern_g4`).

## In this repository

| Path | What |
|---|---|
| `g4numi_ldm/mesons.py` | `macro`: a g4numi run macro from a template (output, seed, run id, POT); `mesons`: unweighted per-species meson lists (pi0, eta, rho0, omega, phi, eta') in BdNMC `particle_list` format with the yield per POT; `protons`: yield per POT of > 100 GeV/c protons at the hadron-absorber planes, per BdNMC bremsstrahlung component |
| `g4numi_ldm/detector.py` | a detector's placement in the target-centred beam frame, from `$LDM_DETECTOR_CONFIG` (placeholder without it) |
| `macros/` | g4numi run macros: medium-energy FHC (`g4numi.mac`, `baseline.mac`) and RHC (`g4numi-RHC.mac`) |
| `mipp/` | the g4numi vs MIPP comparison: `parse_mipp_table.py` (Table I of the paper -> CSV), `compare_g4numi_mipp.py`, `plot_mipp_comparison.py`, `plot_mipp_grid.py`, target-exit and pencil-beam A/B (`analyze_tgtexit.py`, `analyze_ab.py`), `pion_provenance.py`, `outside_target_pions.py`, `plot_pi0_beamline.py`, and the pi0 side (`mipp_pi0_correction.py`, `dm_angular_spread.py`) |

## Requirements

g4numi with the LDM output patches (meson tree, `tgtexit` tree and meson provenance, muon z-planes, optional EM
cascade seeds). The patches are a branch of g4numi; see "g4numi patches" below.

```bash
pip install -e .          # numpy, uproot, awkward, matplotlib
pip install -e '.[pdf]'   # pymupdf, for parse_mipp_table.py
pip install -e '.[test]' && pytest
```

## Use

```bash
python -m g4numi_ldm.mesons macro --template macros/baseline.mac --run-id 3 --pot 1000000 --out run.mac
g4numi run.mac                                           # writes g4numi_nudata_0003.root and zpNtuple_0003.root
python -m g4numi_ldm.mesons mesons g4numi_nudata_0003.root --pot 1000000 --run-id 3 --out-dir particle_lists
python -m g4numi_ldm.mesons protons zpNtuple_0003.root --pot 1000000 --interacting
```

MIPP comparison (the paper, J. M. Paley et al., Phys. Rev. D 90, 032001 (2014), is not redistributed; extract its
Table I from your copy first):

```bash
python mipp/parse_mipp_table.py <paper.pdf> mipp/mipp_table1.csv      # or set MIPP_TABLE
export G4NUMI_RUNS_DIR=<dir with rNNNN/zpNtuple_0NNN.root>
python mipp/compare_g4numi_mipp.py --run r0003
python mipp/plot_mipp_grid.py && python mipp/plot_pi0_beamline.py
LDM_DETECTOR_CONFIG=<detector placement JSON> python mipp/mipp_pi0_correction.py --run r0003 --mchi 0.012
```

| Variable | Default | Content |
|---|---|---|
| `G4NUMI_RUNS_DIR` | `runs` | g4numi runs, `<dir>/rNNNN/zpNtuple_0NNN.root` |
| `G4NUMI_LDM_DATA` | `mipp/data` | slim `.npz` extracts of g4numi runs |
| `G4NUMI_LDM_OUT` | `mipp/plots` | figures and JSON summaries |
| `MIPP_TABLE` | `mipp/mipp_table1.csv` | MIPP Table I as CSV |
| `LDM_DETECTOR_CONFIG` | placeholder | `{"frame": {"origin_m", "pitch_rad", "yaw_rad"}, "detector": {"half_width_m", "length_m"}}` (the `frame` block of a bdnmc-nova configuration also works, via `$BDNMC_NOVA_CONFIG`) |

## Not in this repository, and why

- g4numi outputs (meson lists, ntuples, `.npz` extracts): experiment production data;
- a particular detector's placement (e.g. the NOvA Near Detector frame): experiment configuration, supplied through
  `$LDM_DETECTOR_CONFIG`;
- the MIPP paper's table (published material; regenerate it with `parse_mipp_table.py`);
- comparisons with another analysis's analytic (BMPT) pion source: they use that analysis's inputs.

## g4numi patches

The LDM output patches to g4numi are commits on top of NuSoftHEP/g4numi `main_modern_g4`; g4numi is MIT-licensed
(Fermi National Accelerator Laboratory). They are published, if at all, as a branch of a g4numi fork, not here.

## License

MIT (LICENSE). Third-party material: NOTICE.
