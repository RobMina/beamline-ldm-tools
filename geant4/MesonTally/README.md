# MesonTally — portable Geant4 meson tally

`ldm::MesonTally` writes every neutral meson of the selected PDG codes (default π⁰ 111, η 221, η′ 331, ω 223) to a
ROOT file **when it is stacked**, before the application's stacking action can kill it. Beamline simulations kill
tracks for speed: a kinetic-energy threshold (50 MeV in g4lbne), Russian roulette by importance weight, and the EM
kill. A tally at track start sees only the survivors. Here each meson is recorded with the weight and generation the
application gives the new track, normally the parent's importance weight, so the weighted list is unbiased.

Requirements: Geant4 (tested with 10.4.3) and ROOT 6 (tested with 6.32.06), C++17. No dictionary is needed: plain
TTree branches only.

## Integrate

1. Copy `MesonTally.hh` into the application's include directory and `MesonTally.cc` into its source directory,
   unchanged. Most Geant4 CMake files glob `src/*.cc`.
2. Create one instance before the macro runs (so its UI commands exist), with the application's UI directory and an
   adapter that reads weight and generation from the application's track information:

   ```cpp
   #include "MesonTally.hh"
   ldm::MesonTally* Tally() {
      static ldm::MesonTally* t = new ldm::MesonTally("/LBNE", [](const G4Track* trk) {
         const auto* i = dynamic_cast<const MyTrackInformation*>(trk->GetUserInformation());
         if (i) return ldm::MesonTally::Info{i->GetNImpWt(), i->GetTgen()};
         return ldm::MesonTally::Info{1.0, trk->GetParentID() == 0 ? 1 : -1};
      });
      return t;
   }
   ```
   Call `Tally()` once in the run action's constructor. The optional third argument is the zero-padding of the run
   number in the file name: 5 by default, use 4 to match g4numi's names.
3. Hooks:
   - `BeginOfRunAction(run)`: `Tally()->BeginRun(run->GetRunID());`
   - `EndOfRunAction(run)`: `Tally()->EndRun(run->GetNumberOfEvent(), physicsListName, versionString);`
   - `ClassifyNewTrack(track)`: `Tally()->Consider(track);` as the **first** statement, before any kill.

   The adapter must see the weight the application has already given the new track. In g4lbne (as in g4numi),
   `PostUserTrackingAction` copies the parent's weight onto every secondary before they are stacked.

## Use

```
/LBNE/mesons/FileName mesons        # file = <stem>_<runID>.mesons.root in the working directory
/LBNE/mesons/PDGs 111 221 331 223   # optional
/LBNE/mesons/Create True            # off by default
/run/beamOn ...
```

(with the application's own UI directory in place of `/LBNE`).

## Output

| Tree | Branches |
|---|---|
| `meson` | `px py pz E` [GeV], `x y z` [cm, global], `t` [ns], `weight`, `pdg`, `tgen` (same names and units as the g4numi `meson` tree), `evt`, `trk`, `ppdg` (parent PDG), `pE` [GeV: the parent's total energy at the pre-step point of the step that made the meson, −1 if unknown], `proc`, `vol`, `mat` (ids into `names`) |
| `meta` | `pots` (events in the run), `run`, `nmeson`, `physlist`, `commit` |
| `names` | `id`, `kind` (0 creator process, 1 logical volume, 2 material), `name` |

`python -m beamline_ldm.mesons regions|mesons <file> --beamline <name>` reads it.

## Implementation notes

- A track whose step number is not 0 (a suspended, re-stacked track) is skipped, so nothing is counted twice.
- Parent PDG and energy come from the tracking manager's current track: Geant4 stacks secondaries right after their
  parent's tracking ends, before deleting the parent. A mismatch in track id gives `ppdg = 0`.
- `pE` is read from the pre-step point of the parent's last step, not the parent's current energy. By stacking time
  that is the post-interaction value: the rest mass for a parent killed by its inelastic interaction (fixed in 0.2.1;
  0.2.0 files carry it). It omits only the creating step's continuous loss. A meson made at an earlier step, by a
  parent that survived its interaction, has no recorded step left and gets `pE = -1`.
- Creation volume and material come from a private `G4Navigator`; the tracking navigator is never moved.
- The tally draws no random numbers, so switching it on does not change the simulation. In g4lbnf, the same macro and
  seed gave identical meson files before and after the class was moved here (2,881 entries, 200 POT).
- About 0.8 KB per POT for the LBNF OfficialEng 2021 beamline (14 entries per POT).
