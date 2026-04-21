from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "models" / "frozen_v1" / "manifest.json"


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def repo_path(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


def load_frozen_paths() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "manifest": manifest,
        "segmentation": repo_path(manifest["artifacts"]["segmentation"]["path"]),
        "occupancy": repo_path(manifest["artifacts"]["occupancy"]["path"]),
        "digit": repo_path(manifest["artifacts"]["digit"]["path"]),
        "occ_calibration": repo_path(manifest["artifacts"]["calibration"]["occ"]),
        "digit_calibration": repo_path(manifest["artifacts"]["calibration"]["digit"]),
        "warp_size": manifest["config"]["default_warp_size"],
        "trim_frac": manifest["config"]["trim_frac"],
        "occ_threshold": manifest["config"]["occ_threshold"],
        "readout": manifest["blessed_stack"]["readout"],
    }
