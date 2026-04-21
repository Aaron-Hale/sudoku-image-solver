from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.sudoku_solver.inference import predict_givens_from_image
from src.sudoku_solver.frozen_config import REPO_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Run frozen repo inference on label JSONs and report parity metrics.")
    p.add_argument("--old-repo-root", type=Path, default=Path("/Users/aaronhale/Desktop/sudoku_solver"))
    p.add_argument("--splits", nargs="+", default=["core_test"])
    p.add_argument("--exclude-kaggle", action="store_true", default=False)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "frozen_eval_v1")
    return p.parse_args()


def is_kaggle_record(rec: dict[str, Any]) -> bool:
    return any(str(t).strip().lower() == "kaggle" for t in rec.get("tags", []))


def find_raw_image(old_repo_root: Path, rec: dict[str, Any], split: str) -> Path:
    image_path = rec.get("image_path")
    if isinstance(image_path, str):
        p = old_repo_root / image_path
        if p.exists():
            return p
    image_id = rec["image_id"]
    raw_dir = old_repo_root / "data" / "raw" / split
    for ext in [".jpg", ".jpeg", ".png"]:
        p = raw_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Raw image not found for {image_id}")


def load_records(old_repo_root: Path, splits: list[str], exclude_kaggle: bool) -> list[dict[str, Any]]:
    labels_dir = old_repo_root / "data" / "labels" / "boards"
    out = []
    for jp in sorted(labels_dir.glob("*.json")):
        rec = json.loads(jp.read_text())
        if rec.get("split") not in splits:
            continue
        if exclude_kaggle and is_kaggle_record(rec):
            continue
        out.append(rec)
    return out


def _group_duplicate_excess(vals: list[int]) -> int:
    from collections import Counter
    ctr = Counter([int(v) for v in vals if int(v) > 0])
    return int(sum(max(0, c - 1) for c in ctr.values()))


def grid_legality_metrics(grid: list[list[int]]) -> dict[str, int]:
    row_dup_excess = 0
    col_dup_excess = 0
    box_dup_excess = 0
    for r in range(9):
        row_dup_excess += _group_duplicate_excess(grid[r])
    for c in range(9):
        col_dup_excess += _group_duplicate_excess([grid[r][c] for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = [grid[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)]
            box_dup_excess += _group_duplicate_excess(vals)
    total_dup_excess = int(row_dup_excess + col_dup_excess + box_dup_excess)
    return {
        "row_dup_excess": row_dup_excess,
        "col_dup_excess": col_dup_excess,
        "box_dup_excess": box_dup_excess,
        "total_dup_excess": total_dup_excess,
        "legality_failure": int(total_dup_excess > 0),
    }


def evaluate_pred_grid(pred_grid: list[list[int]], truth_givens: list[list[int]]) -> dict[str, Any]:
    miss_rows = []
    n_givens = 0
    n_true_empty = 0
    n_givens_correct = 0
    n_full_correct = 0
    missed_as_empty = 0
    wrong_digit = 0
    false_positive_on_empty = 0

    for r in range(9):
        for c in range(9):
            truth_digit = int(truth_givens[r][c])
            pred_digit = int(pred_grid[r][c])

            if truth_digit == pred_digit:
                n_full_correct += 1

            if truth_digit > 0:
                n_givens += 1
                if pred_digit == truth_digit:
                    n_givens_correct += 1
                else:
                    failure_type = "missed_as_empty" if pred_digit == 0 else "wrong_digit"
                    if failure_type == "missed_as_empty":
                        missed_as_empty += 1
                    else:
                        wrong_digit += 1
                    miss_rows.append(
                        {
                            "row": r,
                            "col": c,
                            "truth": truth_digit,
                            "pred": pred_digit,
                            "failure_type": failure_type,
                        }
                    )
            else:
                n_true_empty += 1
                if pred_digit > 0:
                    false_positive_on_empty += 1
                    miss_rows.append(
                        {
                            "row": r,
                            "col": c,
                            "truth": 0,
                            "pred": pred_digit,
                            "failure_type": "false_positive_on_empty",
                        }
                    )

    legality = grid_legality_metrics(pred_grid)

    return {
        "n_givens": int(n_givens),
        "n_true_empty": int(n_true_empty),
        "n_givens_correct": int(n_givens_correct),
        "givens_accuracy": float(n_givens_correct / max(1, n_givens)),
        "exact_givens_match": bool(n_givens_correct == n_givens),
        "full_board_cell_accuracy": float(n_full_correct / 81.0),
        "missed_as_empty": int(missed_as_empty),
        "wrong_digit": int(wrong_digit),
        "false_positive_on_empty": int(false_positive_on_empty),
        "pred_nonzero_count": int(sum(int(v > 0) for row in pred_grid for v in row)),
        "miss_rows": miss_rows,
        **legality,
    }


def summarize_results(df: pd.DataFrame, group_cols: list[str], latency_col: str = "runtime_ms") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        row.update(
            {
                "n_eval_boards": int(len(g)),
                "exact_givens_match_rate": float(g["exact_givens_match"].mean()),
                "mean_givens_accuracy": float(g["givens_accuracy"].mean()),
                "mean_full_board_cell_accuracy": float(g["full_board_cell_accuracy"].mean()),
                "total_missed_as_empty": int(g["missed_as_empty"].sum()),
                "total_wrong_digit": int(g["wrong_digit"].sum()),
                "total_false_positive_on_empty": int(g["false_positive_on_empty"].sum()),
                "legality_failure_rate": float(g["legality_failure"].mean()),
                "mean_runtime_ms": float(g[latency_col].mean()),
                "p95_runtime_ms": float(g[latency_col].quantile(0.95)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.old_repo_root, args.splits, exclude_kaggle=args.exclude_kaggle)
    rows = []

    for rec in records:
        raw_path = find_raw_image(args.old_repo_root, rec, rec["split"])
        t0 = time.perf_counter()
        pred = predict_givens_from_image(raw_path)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        metrics = evaluate_pred_grid(pred, rec["givens"])
        rows.append(
            {
                "image_id": rec["image_id"],
                "split": rec["split"],
                "runtime_ms": runtime_ms,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "board_level_rows.csv", index=False)

    split_summary = summarize_results(df, ["split"])
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)

    combined_summary = summarize_results(df.assign(_all="all"), ["_all"]).drop(columns=["_all"])
    combined_summary.to_csv(args.output_dir / "combined_summary.csv", index=False)

    print("=== Split summary ===")
    print(split_summary.to_string(index=False))
    print("\n=== Combined summary ===")
    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
