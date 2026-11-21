# beamline-ldm-tools

Light-dark-matter (LDM) tools for Geant4 **beamline** simulations: recording the neutral mesons a proton beam makes
in its target, horns, decay pipe and absorber, and turning them into the inputs of an LDM generator such as
[BdNMC](https://github.com/RobMina/BdNMC). Experiment-agnostic: beamline specifics are a small backend, and a
downstream detector's placement is a configuration input.

| Beamline | Simulation | Meson output |
|---|---|---|
| `g4numi` | NuMI ([g4numi](https://github.com/NuSoftHEP/g4numi)) | `meson` tree from the `ldm` branch of [RobMina/g4numi](https://github.com/RobMina/g4numi/tree/ldm) (`g4numi_nudata_NNNN.root`) |
| `g4lbnf` | LBNF (g4lbne / g4lbnf) | `geant4/MesonTally` vendored into g4lbne (`<stem>_NNNNN.mesons.root`) |

Formerly `g4numi-ldm-tools` (renamed 2026-11-21); the `g4numi_ldm` import name and `python -m g4numi_ldm.mesons`
still work and select the g4numi backend.

## In this repository

| Path | What |
|---|---|
| `geant4/MesonTally/` | **portable Geant4 meson tally** (`ldm::MesonTally`, C++, ROOT output): records every π⁰/η/η′/ω (configurable) when it is stacked, before an application's energy-threshold and importance-weight kills, with weight, generation, parent, creator process and creation volume/material. Vendor it into any Geant4 application; [integration guide](geant4/MesonTally/README.md) |
| `beamline_ldm/mesons.py` | `macro`: a run macro from a template (output, seed, run id, POT); `mesons`: unweighted per-species meson lists (pi0, eta, rho0, omega, phi, eta') in BdNMC `particle_list` format with the yield per POT, optionally split by source region; `regions`: weighted yield per POT by species and region; `protons`: g4numi yield per POT of > 100 GeV/c protons at the hadron-absorber planes, per BdNMC bremsstrahlung component |
| `beamline_ldm/beamlines.py` | the beamline backends: POT metadata tree, target centre (origin of the list frame), source-region rule, macro directory |
| `beamline_ldm/detector.py` | a detector's placement in the target-centred beam frame, from `$LDM_DETECTOR_CONFIG` (placeholder without it) |
| `macros/` | g4numi run macros: medium-energy FHC (`g4numi.mac`, `baseline.mac`) and RHC (`g4numi-RHC.mac`) |
| `mipp/` | the g4numi vs MIPP comparison: `parse_mipp_table.py` (Table I of the paper -> CSV), `compare_g4numi_mipp.py`, `plot_mipp_comparison.py`, `plot_mipp_grid.py`, target-exit and pencil-beam A/B (`analyze_tgtexit.py`, `analyze_ab.py`), `pion_provenance.py`, `outside_target_pions.py`, `plot_pi0_beamline.py`, and the pi0 side (`mipp_pi0_correction.py`, `dm_angular_spread.py`) |

## The meson tree

All backends read one schema: `px py pz E` (GeV), `x y z` (cm, the simulation's global frame), `t` (ns), `weight`,
`pdg`, `tgen`. `MesonTally` adds `evt trk ppdg pE proc vol mat` and the `meta` (`pots`, `run`, `nmeson`, `physlist`,
`commit`) and `names` (id legend for `proc`/`vol`/`mat`) trees.

Lists are written in the target-centred frame (z minus the backend's target centre), metres and seconds:
`px py pz E x y z t`, with `yield_per_POT=` in the header. Resampling uses `np.random.seed(42 + run id)` (legacy
global seeding: reproduces the NOvA production lists bit for bit).

### Source regions

| Backend | Regions | Rule |
|---|---|---|
| `g4numi` | `upstream`, `absorber` | z (g4numi cm) at or beyond the pre-absorber concrete face, 72183.4 cm, is `absorber`; `upstream` is target + horns + decay pipe. |
| `g4lbnf` | `target`, `horns`, `decaypipe`, `absorber`, `rock` | from `MesonTally`'s creation volume/material: graphite (`Target` material) = target; `*DecayPipe*` volumes; `RockLogical`; anything else beyond z = 218 m = absorber; the rest upstream = horns |

Region yields come from the unfiltered pool, so they add up to the total.

## Requirements

```bash
pip install -e .          # numpy, uproot, awkward, matplotlib
pip install -e '.[pdf]'   # pymupdf, for parse_mipp_table.py
pip install -e '.[test]' && pytest
```

## Use

```bash
# NuMI (g4numi, the default backend)
python -m beamline_ldm.mesons macro --template macros/baseline.mac --run-id 3 --pot 1000000 --out run.mac
g4numi run.mac                                           # writes g4numi_nudata_0003.root and zpNtuple_0003.root
python -m beamline_ldm.mesons mesons g4numi_nudata_0003.root --pot 1000000 --run-id 3 --out-dir particle_lists
python -m beamline_ldm.mesons protons zpNtuple_0003.root --pot 1000000 --interacting

# LBNF (g4lbnf with MesonTally); POT comes from the file's meta tree
python -m beamline_ldm.mesons macro --beamline g4lbnf --template OfficialEngDesignSept2021.mac --run-id 11 \
    --pot 200000 --stem lbnf --out run.mac
g4lbnf run.mac                                           # writes lbnf_00011.mesons.root (+ dk2nu)
python -m beamline_ldm.mesons regions lbnf_00011.mesons.root --beamline g4lbnf
python -m beamline_ldm.mesons mesons lbnf_00011.mesons.root --beamline g4lbnf --run-id 11 --split-regions
```

Adding a beamline: register a `Beamline` (POT tree, target centre, region rule, macro directory) in
`beamline_ldm/beamlines.py`, give `mesons.write_macro` its commands, and vendor `geant4/MesonTally` into the
simulation.

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

- beamline-simulation outputs (meson lists, ntuples, `.npz` extracts): experiment production data;
- a particular detector's placement (e.g. the NOvA Near Detector frame): experiment configuration, supplied through
  `$LDM_DETECTOR_CONFIG`;
- the MIPP paper's table (published material; regenerate it with `parse_mipp_table.py`);
- comparisons with another analysis's analytic (BMPT) pion source: they use that analysis's inputs.

## Patches to the simulations

The LDM output patches to g4numi are commits on top of NuSoftHEP/g4numi `main_modern_g4`; g4numi is MIT-licensed
(Fermi National Accelerator Laboratory). They are published as the `ldm` branch of a g4numi fork, not here (they
predate `MesonTally`, record mesons at track start rather than at stacking, and have not been moved onto it).

For g4lbne only the experiment-agnostic part is here (`geant4/MesonTally`). The g4lbne glue (an adapter for its
track information plus four one-line hooks, see the integration guide) is not redistributed with g4lbne sources.

## License

MIT (LICENSE). Third-party material: NOTICE.
