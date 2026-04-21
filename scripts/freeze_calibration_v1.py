#!/usr/bin/env python3
"""
Export the final NB20 calibration layer and patch the frozen model manifest.

Source notebook provenance:
- 20_day28_calibration_constrained_decode_v5.ipynb

Canonical notebook-derived final readout:
- blessed variant: occ_platt_digit_temp_no_decode
- occ_threshold: 0.35
- occupancy calibration: platt(a=0.7386137843132019, b=-0.34071943163871765)
- digit calibration: temp(t=0.6822174787521362)

Important:
- In NB20, SAVE_NOTEBOOK_ARTIFACTS=False, so nothing was written.
- This script exists to export the notebook-canonical in-memory readout into
  repo-tracked JSON files for reproducibility/governance.
- This script does NOT refit calibrators; it freezes the final notebook-derived values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


NOTEBOOK_SOURCE = "20_day28_calibration_constrained_decode_v5.ipynb"

# Canonical notebook-derived final values from NB20 v5
BLESSED_VARIANT = "occ_platt_digit_temp_no_decode"
OCC_THRESHOLD = 0.35

OCC_CAL = {
    "kind": "platt",
    "a": 0.7386137843132019,
    "b": -0.34071943163871765,
}
DIGIT_CAL = {
    "kind": "temp",
    "t": 0.6822174787521362,
}

NOTEBOOK_CONFIG = {
    "train_split": "core_train",
    "eval_splits": ["core_val", "core_test"],
    "include_kaggle_train": False,
    "exclude_kaggle_eval": True,
    "default_warp_size": 900,
    "trim_frac": 0.12,
    "baseline_occ_threshold": 0.50,
    "occ_cal_epochs": 600,
    "digit_cal_epochs": 800,
    "cal_lr": 5e-2,
    "weight_decay": 1e-4,
    "occ_threshold_grid_start": 0.35,
    "occ_threshold_grid_stop": 0.75,
    "occ_threshold_grid_step": 0.025,
}

FROZEN_METRICS = {
    "combined_non_kaggle": {
        "n_eval_boards": 121,
        "exact_givens_match_rate": 0.859504,
        "mean_givens_accuracy": 0.975222,
        "mean_full_board_cell_accuracy": 0.988369,
        "total_missed_as_empty": 71,
        "total_wrong_digit": 22,
        "total_false_positive_on_empty": 21,
        "legality_failure_rate": 0.082645,
        "mean_total_est_latency_ms": 230.852254,
        "median_total_est_latency_ms": 230.943708,
        "p95_total_est_latency_ms": 234.624667,
    },
    "split_breakdown": {
        "core_test": {
            "n_eval_boards": 62,
            "exact_givens_match_rate": 0.887097,
            "mean_givens_accuracy": 0.985379,
            "mean_full_board_cell_accuracy": 0.994225,
            "total_missed_as_empty": 22,
            "total_wrong_digit": 5,
        },
        "core_val": {
            "n_eval_boards": 59,
            "exact_givens_match_rate": 0.830508,
            "mean_givens_accuracy": 0.964549,
            "mean_full_board_cell_accuracy": 0.982214,
            "total_missed_as_empty": 49,
            "total_wrong_digit": 17,
        },
    },
    "latency_benchmark": {
        "in_memory": {
            "mean_total_ms": 229.912455,
            "median_total_ms": 229.936708,
            "p95_total_ms": 233.135758,
            "mean_seg_and_corners_ms": 218.581451,
            "mean_warp_and_crop_ms": 0.985275,
            "mean_model_infer_ms": 10.287473,
            "mean_calibration_decode_ms": 0.058256,
        },
        "full_hot_path": {
            "mean_read_decode_ms": 4.659947,
            "mean_infer_ms": 228.383459,
            "mean_full_ms": 233.206886,
            "median_full_ms": 232.948833,
            "p95_full_ms": 239.557425,
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze the final NB20 calibration layer into repo-tracked JSON files.")
    p.add_argument("--repo-root", type=Path, default=Path("/Users/aaronhale/projects/sudoku-image-solver"))
    p.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/models/frozen_v1/calibration",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/models/frozen_v1/manifest.json",
    )
    p.add_argument("--patch-manifest", action="store_true", help="Patch manifest.json with calibration file paths and occ_threshold.")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    calibration_dir = args.calibration_dir or (args.repo_root / "models" / "frozen_v1" / "calibration")
    manifest_path = args.manifest_path or (args.repo_root / "models" / "frozen_v1" / "manifest.json")

    ensure_dir(calibration_dir)

    occ_path = calibration_dir / "occ_calibration.json"
    digit_path = calibration_dir / "digit_calibration.json"
    calib_manifest_path = calibration_dir / "calibration_manifest.json"

    write_json(occ_path, OCC_CAL)
    write_json(digit_path, DIGIT_CAL)
    write_json(
        calib_manifest_path,
        {
            "source_notebook": NOTEBOOK_SOURCE,
            "note": "NB20 had SAVE_NOTEBOOK_ARTIFACTS=False, so these JSON files are the exported canonical notebook-derived values.",
            "blessed_variant": BLESSED_VARIANT,
            "occ_threshold": OCC_THRESHOLD,
            "occ_calibration_path": str(occ_path),
            "digit_calibration_path": str(digit_path),
            "notebook_config": NOTEBOOK_CONFIG,
            "frozen_metrics": FROZEN_METRICS,
        },
    )

    print(f"Wrote: {occ_path}")
    print(f"Wrote: {digit_path}")
    print(f"Wrote: {calib_manifest_path}")

    if args.patch_manifest:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)

        manifest.setdefault("blessed_stack", {})
        manifest["blessed_stack"]["readout"] = BLESSED_VARIANT

        manifest.setdefault("artifacts", {})
        manifest["artifacts"].setdefault("calibration", {})
        manifest["artifacts"]["calibration"]["occ"] = str(occ_path.relative_to(args.repo_root))
        manifest["artifacts"]["calibration"]["digit"] = str(digit_path.relative_to(args.repo_root))

        manifest.setdefault("config", {})
        manifest["config"]["occ_threshold"] = OCC_THRESHOLD

        # Optionally refresh the official metrics if present.
        manifest.setdefault("official_metrics", {})
        manifest["official_metrics"]["combined_non_kaggle_exact_givens"] = FROZEN_METRICS["combined_non_kaggle"]["exact_givens_match_rate"]
        manifest["official_metrics"]["core_test_non_kaggle_exact_givens"] = FROZEN_METRICS["split_breakdown"]["core_test"]["exact_givens_match_rate"]
        manifest["official_metrics"]["hot_latency_mean_ms"] = FROZEN_METRICS["latency_benchmark"]["full_hot_path"]["mean_full_ms"]
        manifest["official_metrics"]["hot_latency_p95_ms"] = FROZEN_METRICS["latency_benchmark"]["full_hot_path"]["p95_full_ms"]

        write_json(manifest_path, manifest)
        print(f"Patched manifest: {manifest_path}")


if __name__ == "__main__":
    main()
