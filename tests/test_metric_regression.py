from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("SUDOKU_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    not DATA_ROOT,
    reason="Set SUDOKU_DATA_ROOT to run eval regression tests.",
)


def run_eval(*splits: str) -> str:
    cmd = [
        "python",
        "scripts/run_frozen_eval_v1.py",
        "--data-root",
        DATA_ROOT,
        "--splits",
        *splits,
        "--exclude-kaggle",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
    )
    return proc.stdout


def extract_combined_metric(output: str, metric: str) -> float:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    idx = lines.index("=== Combined summary ===")
    header = re.split(r"\s{2,}", lines[idx + 1].strip())
    values = re.split(r"\s{2,}", lines[idx + 2].strip())
    row = dict(zip(header, values))
    return float(row[metric])


def test_core_test_metric_regression():
    out = run_eval("core_test")
    assert abs(extract_combined_metric(out, "exact_givens_match_rate") - 0.887097) < 1e-6
    assert abs(extract_combined_metric(out, "mean_givens_accuracy") - 0.985379) < 1e-6
    assert abs(extract_combined_metric(out, "mean_full_board_cell_accuracy") - 0.994225) < 1e-6
    assert abs(extract_combined_metric(out, "legality_failure_rate") - 0.048387) < 1e-6


def test_combined_metric_regression():
    out = run_eval("core_val", "core_test")
    assert abs(extract_combined_metric(out, "exact_givens_match_rate") - 0.859504) < 1e-6
    assert abs(extract_combined_metric(out, "mean_givens_accuracy") - 0.975222) < 1e-6
    assert abs(extract_combined_metric(out, "mean_full_board_cell_accuracy") - 0.988369) < 1e-6
    assert abs(extract_combined_metric(out, "legality_failure_rate") - 0.082645) < 1e-6
