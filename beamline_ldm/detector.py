"""A downstream detector's placement in the target-centred beam frame, read from configuration.

No experiment's values are in this repository. The JSON file named by `$LDM_DETECTOR_CONFIG` (or, for the frame
only, `$BDNMC_NOVA_CONFIG`, whose "frame" block has the same format) holds

    {"frame": {"origin_m": [x0, y0, z0], "pitch_rad": a, "yaw_rad": b},
     "detector": {"half_width_m": w, "length_m": l}}

with beam = X0 + R det, R built from a pitch (about x) and a yaw (about y), the convention of bdnmc-nova. Without a
file the placeholder below (an on-axis box 1 km downstream) is used, with a warning.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np

PLACEHOLDER = {"frame": {"origin_m": [0.0, 0.0, 1000.0], "pitch_rad": 0.0, "yaw_rad": 0.0},
               "detector": {"half_width_m": 2.0, "length_m": 10.0}}


def axes(pitch: float, yaw: float) -> np.ndarray:
    """R: the detector's x, y, z axes (columns) in the beam frame."""
    ca, sa, cb, sb = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cb, -sb * sa, sb * ca], [0.0, ca, sa], [-sb, -cb * sa, cb * ca]])


def load(path: str | None = None) -> dict:
    cfg = {}
    path = path or os.environ.get("LDM_DETECTOR_CONFIG")
    if path:
        cfg = json.loads(Path(path).read_text())
    elif os.environ.get("BDNMC_NOVA_CONFIG"):
        cfg = {"frame": json.loads(Path(os.environ["BDNMC_NOVA_CONFIG"]).read_text())["frame"]}
    else:
        warnings.warn("no $LDM_DETECTOR_CONFIG: using a placeholder detector placement", stacklevel=2)
    out = {k: dict(v) for k, v in PLACEHOLDER.items()}
    for k in out:
        out[k].update(cfg.get(k, {}))
    return out
