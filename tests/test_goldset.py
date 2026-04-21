from __future__ import annotations

import json
from pathlib import Path

from src.sudoku_solver.frozen_config import REPO_ROOT, load_frozen_paths
from src.sudoku_solver.inference import predict_givens_from_image


GOLD_MANIFEST = REPO_ROOT / "tests" / "goldset" / "manifests" / "gold_regression.jsonl"


def load_gold_rows() -> list[dict]:
    rows: list[dict] = []
    with open(GOLD_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_gold_manifest_is_valid():
    rows = load_gold_rows()
    assert len(rows) == 12, f"Expected 12 gold rows, found {len(rows)}"

    seen_ids = set()
    for row in rows:
        assert "image_id" in row
        assert "split" in row
        assert "image_path" in row
        assert "expected_givens" in row

        assert row["split"] == "core_test"
        assert row["image_id"] not in seen_ids
        seen_ids.add(row["image_id"])

        img_path = REPO_ROOT / row["image_path"]
        assert img_path.exists(), f"Missing image: {img_path}"

        givens = row["expected_givens"]
        assert isinstance(givens, list) and len(givens) == 9
        for r in givens:
            assert isinstance(r, list) and len(r) == 9
            for x in r:
                assert isinstance(x, int)
                assert 0 <= x <= 9


def test_frozen_manifest_paths_exist():
    frozen = load_frozen_paths()

    assert frozen["segmentation"].exists()
    assert frozen["occupancy"].exists()
    assert frozen["digit"].exists()
    assert frozen["occ_calibration"].exists()
    assert frozen["digit_calibration"].exists()

    assert frozen["warp_size"] == 900
    assert abs(float(frozen["trim_frac"]) - 0.12) < 1e-12
    assert abs(float(frozen["occ_threshold"]) - 0.35) < 1e-12
    assert frozen["readout"] == "occ_platt_digit_temp_no_decode"


def test_gold_predictions_match_expected():
    rows = load_gold_rows()
    for row in rows:
        image_path = REPO_ROOT / row["image_path"]
        pred = predict_givens_from_image(image_path)
        assert pred == row["expected_givens"], f"Mismatch for {row['image_id']}"
