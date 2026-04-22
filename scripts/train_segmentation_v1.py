#!/usr/bin/env python3
"""
Reproduce the NB15 segmentation retraining decision and export frozen segmentation artifacts.

Source notebook provenance:
- 15_stagewise_combined_inline_eval.ipynb
- NB15 trained both stretch and letterbox segmentation models inside the notebook,
  then saved them via the cell:
  "Save trained segmentation models so you do not have to retrain after a restart."

Frozen notebook configs mirrored here:
- RANDOM_SEED = 42
- SEG_TRAIN_IMAGE_SIZE = 768
- SEG_BASE_CHANNELS = 48
- SEG_BATCH_SIZE = 4
- SEG_EPOCHS = 35
- SEG_LR = 1e-3
- SEG_WEIGHT_DECAY = 1e-4
- SEG_NUM_WORKERS = 0
- train data = non-Kaggle core_train only
- internal segmentation val = 10% holdout from non-Kaggle core_train, minimum 20 boards

Outputs:
- reports/stagewise_eval_inline/_saved_seg_models/stretch_seg_checkpoint.pt
- reports/stagewise_eval_inline/_saved_seg_models/letterbox_seg_checkpoint.pt
- reports/stagewise_eval_inline/seg_retrain_summary.json
- reports/stagewise_eval_inline/seg_retrain_history.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain the NB15 segmentation models and export frozen artifacts.")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--train-split", default="core_train")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--min-holdout", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=768)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--modes", nargs="+", default=["stretch", "letterbox"], choices=["stretch", "letterbox"])
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/reports/stagewise_eval_inline/_saved_seg_models",
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/reports/stagewise_eval_inline/seg_retrain_summary.json",
    )
    p.add_argument(
        "--history-csv-path",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/reports/stagewise_eval_inline/seg_retrain_history.csv",
    )
    return p.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_board_labels(labels_dir: Path, split: str) -> List[Dict]:
    rows: List[Dict] = []
    for jp in sorted(labels_dir.glob("*.json")):
        data = load_json(jp)
        if data.get("split") == split:
            rows.append(data)
    return rows


def _record_contains_kaggle_tag(obj) -> bool:
    if isinstance(obj, dict):
        return any(_record_contains_kaggle_tag(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_record_contains_kaggle_tag(v) for v in obj)
    if isinstance(obj, str):
        s = obj.strip().lower()
        return s == "kaggle" or "kaggle" in s
    return False


def is_kaggle_record(rec: Dict) -> bool:
    return _record_contains_kaggle_tag(rec)


def find_raw_image(repo_root: Path, rec: Dict, split: str) -> Path:
    image_path = rec.get("image_path")
    if isinstance(image_path, str):
        p = repo_root / image_path
        if p.exists():
            return p
    image_id = rec["image_id"]
    for ext in [".jpg", ".jpeg", ".png"]:
        p = repo_root / "data" / "raw" / split / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Raw image not found for {image_id}")


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def resize_points(points: np.ndarray, orig_w: int, orig_h: int, new_size: int) -> np.ndarray:
    out = points.copy().astype(np.float32)
    out[:, 0] = out[:, 0] * (new_size / orig_w)
    out[:, 1] = out[:, 1] * (new_size / orig_h)
    return out


def make_board_mask(points_tl_tr_br_bl: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = order_points(points_tl_tr_br_bl).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def letterbox_bgr(img_bgr: np.ndarray, target_size: int):
    h, w = img_bgr.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=img_bgr.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    meta = {
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale": scale,
        "target_size": target_size,
    }
    return canvas, meta


def corners_to_letterbox(corners_xy: np.ndarray, meta: Dict) -> np.ndarray:
    pts = np.asarray(corners_xy, dtype=np.float32).copy()
    pts[:, 0] = pts[:, 0] * meta["scale"] + meta["pad_x"]
    pts[:, 1] = pts[:, 1] * meta["scale"] + meta["pad_y"]
    return pts


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin: int, skip_c: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(cin + skip_c, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BoardUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class SegBoardDataset(Dataset):
    def __init__(self, repo_root: Path, items: List[Dict], image_size: int, mode: str, train: bool):
        self.repo_root = repo_root
        self.items = items
        self.image_size = image_size
        self.mode = mode
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        rec = item["rec"]
        split = item["split"]

        raw_path = find_raw_image(self.repo_root, rec, split)
        raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise FileNotFoundError(raw_path)

        gt_corners = order_points(np.array([
            rec["corners"]["top_left"],
            rec["corners"]["top_right"],
            rec["corners"]["bottom_right"],
            rec["corners"]["bottom_left"],
        ], dtype=np.float32))

        h, w = raw_bgr.shape[:2]
        if self.mode == "stretch":
            img = cv2.resize(raw_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            pts = resize_points(gt_corners, w, h, self.image_size)
        elif self.mode == "letterbox":
            img, lb_meta = letterbox_bgr(raw_bgr, self.image_size)
            pts = corners_to_letterbox(gt_corners, lb_meta)
        else:
            raise ValueError(self.mode)

        mask = make_board_mask(pts, self.image_size)

        if self.train:
            if np.random.rand() < 0.7:
                alpha = float(np.random.uniform(0.9, 1.1))
                beta = float(np.random.uniform(-15, 15))
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                k = int(np.random.choice([3, 5]))
                img = cv2.GaussianBlur(img, (k, k), 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = (mask.astype(np.float32) / 255.0)[None, :, :]

        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask)}


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def batch_iou(logits, targets, thr: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return float(((inter + 1e-6) / (union + 1e-6)).mean().item())


def run_seg_epoch(model, loader, device: str, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    iou_sum = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item())
            iou_sum += batch_iou(logits, y)
            n_batches += 1

    return {"loss": loss_sum / max(n_batches, 1), "iou": iou_sum / max(n_batches, 1)}


def build_non_kaggle_items(repo_root: Path, split: str) -> List[Dict]:
    labels_dir = repo_root / "data" / "labels" / "boards"
    items: List[Dict] = []
    for rec in load_board_labels(labels_dir, split):
        if not is_kaggle_record(rec):
            items.append({"split": split, "rec": rec})
    return items


def split_train_and_holdout(items: List[Dict], seed: int, holdout_frac: float, min_holdout: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    val_count = max(min_holdout, int(round(holdout_frac * len(items))))
    holdout_idx = set(perm[:val_count].tolist())
    train_items = [items[i] for i in range(len(items)) if i not in holdout_idx]
    holdout_items = [items[i] for i in range(len(items)) if i in holdout_idx]
    return train_items, holdout_items


def train_one_mode(
    repo_root: Path,
    train_items: List[Dict],
    val_items: List[Dict],
    args: argparse.Namespace,
    seg_mode: str,
    device: str,
):
    train_ds = SegBoardDataset(repo_root, train_items, image_size=args.image_size, mode=seg_mode, train=True)
    val_ds = SegBoardDataset(repo_root, val_items, image_size=args.image_size, mode=seg_mode, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BoardUNet(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict] = []
    best_val_iou = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_out = run_seg_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_out = run_seg_epoch(model, val_loader, device=device, optimizer=None)
        row = {
            "seg_mode": seg_mode,
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "train_iou": train_out["iou"],
            "val_loss": val_out["loss"],
            "val_iou": val_out["iou"],
        }
        history.append(row)
        if val_out["iou"] > best_val_iou:
            best_val_iou = val_out["iou"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            print(
                f"{seg_mode} | epoch={epoch:02d} "
                f"train_loss={train_out['loss']:.4f} train_iou={train_out['iou']:.4f} "
                f"val_loss={val_out['loss']:.4f} val_iou={val_out['iou']:.4f}"
            )

    best_model = BoardUNet(in_channels=3, base_channels=args.base_channels).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    return {
        "seg_mode": seg_mode,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "best_val_iou": best_val_iou,
        "history": history,
        "model": best_model,
    }


def main() -> None:
    args = parse_args()
    device = get_device()

    save_dir = args.save_dir or (args.repo_root / "reports" / "stagewise_eval_inline" / "_saved_seg_models")
    summary_path = args.summary_path or (args.repo_root / "reports" / "stagewise_eval_inline" / "seg_retrain_summary.json")
    history_csv_path = args.history_csv_path or (args.repo_root / "reports" / "stagewise_eval_inline" / "seg_retrain_history.csv")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train_items_full = build_non_kaggle_items(args.repo_root, args.train_split)
    print(f"Non-Kaggle {args.train_split} boards:", len(train_items_full))
    seg_train_items, seg_internal_val_items = split_train_and_holdout(
        train_items_full,
        seed=args.random_seed,
        holdout_frac=args.holdout_frac,
        min_holdout=args.min_holdout,
    )
    print("Seg train items       :", len(seg_train_items))
    print("Seg internal val items:", len(seg_internal_val_items))
    print("Using device          :", device)

    ensure_dir(save_dir)
    all_history_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for seg_mode in args.modes:
        print("\n====================")
        print(f"Training seg model: {seg_mode}")
        print("====================")
        bundle = train_one_mode(
            repo_root=args.repo_root,
            train_items=seg_train_items,
            val_items=seg_internal_val_items,
            args=args,
            seg_mode=seg_mode,
            device=device,
        )
        save_path = save_dir / f"{seg_mode}_seg_checkpoint.pt"
        torch.save(
            {
                "seg_mode": bundle["seg_mode"],
                "image_size": bundle["image_size"],
                "base_channels": bundle["base_channels"],
                "best_val_iou": bundle["best_val_iou"],
                "history": bundle["history"],
                "model_state": bundle["model"].state_dict(),
            },
            save_path,
        )
        print(f"Saved: {save_path}")
        summary_rows.append(
            {
                "seg_mode": bundle["seg_mode"],
                "checkpoint_path": str(save_path),
                "best_val_iou": bundle["best_val_iou"],
                "image_size": bundle["image_size"],
                "base_channels": bundle["base_channels"],
                "epochs": len(bundle["history"]),
                "random_seed": args.random_seed,
                "holdout_frac": args.holdout_frac,
                "min_holdout": args.min_holdout,
                "train_items": len(seg_train_items),
                "internal_val_items": len(seg_internal_val_items),
            }
        )
        all_history_rows.extend(bundle["history"])

    write_json(
        summary_path,
        {
            "source_notebook": "15_stagewise_combined_inline_eval.ipynb",
            "frozen_notebook_configs": {
                "random_seed": args.random_seed,
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "holdout_frac": args.holdout_frac,
                "min_holdout": args.min_holdout,
                "train_split": args.train_split,
                "non_kaggle_only": True,
            },
            "summary_rows": summary_rows,
        },
    )
    write_csv(history_csv_path, all_history_rows)
    print("Wrote summary:", summary_path)
    print("Wrote history:", history_csv_path)


if __name__ == "__main__":
    main()
