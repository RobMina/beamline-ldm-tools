"""Beamline backends: g4lbnf files (meta + names trees), region splits, macro writers, CLI."""
import json

import numpy as np
import pytest
import uproot

from beamline_ldm import beamlines, mesons


def _g4lbnf_file(path, n_target=30, n_absorber=10, n_pipe=5):
    """A tiny MesonTally-format file: pi0 in the target and decay pipe, pi0 + eta in the absorber."""
    n = n_target + n_absorber + n_pipe
    rng = np.random.default_rng(1)
    z = np.concatenate([np.full(n_target, 75.0), np.full(n_absorber, 22600.0), np.full(n_pipe, 10000.0)])
    pdg = np.full(n, 111, dtype=np.int32)
    pdg[n_target:n_target + 3] = 221
    # names: proc 0 = protonInelastic; vol 0 tCore1Log, 1 Sld_Al_1, 2 DecayPipeWall; mat 0 Target, 1 AL, 2 Steel316
    vol = np.concatenate([np.zeros(n_target), np.ones(n_absorber), np.full(n_pipe, 2)]).astype(np.int32)
    mat = np.concatenate([np.zeros(n_target), np.ones(n_absorber), np.full(n_pipe, 2)]).astype(np.int32)
    with uproot.recreate(path) as f:
        f["meson"] = {"px": rng.normal(size=n), "py": rng.normal(size=n), "pz": rng.uniform(1, 10, n),
                      "E": rng.uniform(1, 10, n), "x": np.zeros(n), "y": np.zeros(n), "z": z, "t": np.ones(n),
                      "weight": np.full(n, 2.0), "pdg": pdg, "tgen": np.full(n, 2, dtype=np.int32),
                      "vol": vol, "mat": mat}
        f["meta"] = {"pots": np.array([20], dtype=np.int32), "run": np.array([11], dtype=np.int32)}
        f["names"] = {"id": np.array([0, 0, 1, 2, 0, 1, 2], dtype=np.int32),
                      "kind": np.array([0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
                      "name": ["protonInelastic", "tCore1Log", "Sld_Al_1", "DecayPipeWall", "Target", "AL",
                               "Steel316"]}
    return path


def test_registry():
    assert set(beamlines.BEAMLINES) >= {"g4numi", "g4lbnf"}
    with pytest.raises(KeyError):
        beamlines.get("nope")


def test_g4lbnf_pot_and_regions(tmp_path):
    f = _g4lbnf_file(tmp_path / "lbnf_00011.mesons.root")
    assert mesons.simulated_pot(f, 0.0, "g4lbnf") == 20.0
    y = mesons.region_yields(f, 20.0, "g4lbnf")
    assert np.isclose(y["pi0"]["target"], 30 * 2.0 / 20) and np.isclose(y["pi0"]["decaypipe"], 5 * 2.0 / 20)
    assert np.isclose(y["pi0"]["absorber"], 7 * 2.0 / 20) and np.isclose(y["eta"]["absorber"], 3 * 2.0 / 20)
    assert np.isclose(sum(v for k, v in y["pi0"].items() if k != "total"), y["pi0"]["total"])


def test_g4lbnf_lists_and_frame(tmp_path):
    f = _g4lbnf_file(tmp_path / "lbnf_00011.mesons.root")
    y = mesons.extract_mesons(f, tmp_path / "lists", run_id=11, pots=20.0, beamline="g4lbnf", region="target")
    assert np.isclose(y["pi0"], 3.0) and y["eta"] == 0.0
    rows = np.loadtxt(tmp_path / "lists" / "pi0_list_0011_target.dat")
    assert rows.shape == (30, 8) and np.allclose(rows[:, 6], 0.0)   # z relative to the 75 cm target centre


def test_g4numi_regions_by_z():
    a = {"z": np.array([-81.65, 50000.0, 72183.4, 72500.0])}
    assert list(beamlines.get("g4numi").regions(a, None)) == ["upstream", "upstream", "absorber", "absorber"]


def test_g4lbnf_macro(tmp_path):
    tpl = tmp_path / "t.mac"
    tpl.write_text("/LBNE/det/construct\n/run/initialize\n/LBNE/run/setRunID 1\n/LBNE/run/NEvents 5000\n"
                   "/run/beamOn\nexit\n")
    text = mesons.write_macro(tpl, tmp_path / "run.mac", 7, 100, beamline="g4lbnf", stem="lbnf").read_text()
    assert "/LBNE/run/NEvents 100" in text and "/LBNE/run/setRunID 7" in text and "NEvents 5000" not in text
    assert text.index("/LBNE/mesons/Create True") < text.index("/run/beamOn") and text.count("/run/beamOn") == 1


def test_cli_split_regions(tmp_path, capsys):
    f = _g4lbnf_file(tmp_path / "lbnf_00011.mesons.root")
    mesons.main(["mesons", str(f), "--beamline", "g4lbnf", "--run-id", "11", "--out-dir", str(tmp_path / "l"),
                 "--split-regions"])
    res = json.loads(capsys.readouterr().out)
    assert set(res) == {"all", "absorber", "decaypipe", "target"}
    assert (tmp_path / "l" / "pi0_list_0011.dat").exists() and (tmp_path / "l" / "eta_list_0011_absorber.dat").exists()
