"""Macro generation and meson-list extraction on a tiny synthetic meson tree."""
import numpy as np
import uproot

from g4numi_ldm import detector, mesons


def test_write_macro(tmp_path):
    tpl = tmp_path / "t.mac"
    tpl.write_text("/NuMI/det/BeamConfig me000z200i\n/run/beamOn 5\n/NuMI/run/setRunID 1\n")
    text = mesons.write_macro(tpl, tmp_path / "run.mac", run_id=7, pot=100).read_text()
    assert "/run/beamOn 100" in text and "/NuMI/run/setRunID 7" in text and "beamOn 5" not in text
    assert "BeamConfig me000z200i" in text


def test_extract_mesons(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    data = {"px": rng.normal(size=n), "py": rng.normal(size=n), "pz": rng.uniform(1, 10, n),
            "E": rng.uniform(1, 10, n), "x": np.zeros(n), "y": np.zeros(n),
            "z": np.full(n, mesons.TARGET_CENTER_Z_G4NUMI), "t": np.full(n, 1.0),
            "weight": np.full(n, 2.0), "pdg": np.where(np.arange(n) < 40, 111, 221).astype(np.int32)}
    f = tmp_path / "m.root"
    with uproot.recreate(f) as out:
        out["meson"] = data
    y = mesons.extract_mesons(f, tmp_path / "lists", run_id=3, pots=10.0)
    assert np.isclose(y["pi0"], 40 * 2.0 / 10.0) and np.isclose(y["eta"], 10 * 2.0 / 10.0) and y["rho0"] == 0.0
    rows = np.loadtxt(tmp_path / "lists" / "pi0_list_0003.dat")
    assert rows.shape == (40, 8) and np.allclose(rows[:, 6], 0.0) and np.allclose(rows[:, 7], 1e-9)


def test_detector_placeholder(monkeypatch):
    monkeypatch.delenv("LDM_DETECTOR_CONFIG", raising=False)
    monkeypatch.delenv("BDNMC_NOVA_CONFIG", raising=False)
    cfg = detector.load()
    assert cfg["frame"]["origin_m"] == [0.0, 0.0, 1000.0]
    assert np.allclose(detector.axes(0.0, 0.0), np.eye(3))
