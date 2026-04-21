from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from src.sudoku_solver.frozen_config import load_frozen_paths


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


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


class SmallDigitCNN(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_classes: int = 9,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        pool_hw: int = 4,
        hidden_dim: int = 256,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pool_hw, pool_hw)),
        )
        flat_dim = c3 * pool_hw * pool_hw
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def infer_digit_cnn_kwargs_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    c1 = int(state_dict["features.0.weight"].shape[0])
    c2 = int(state_dict["features.3.weight"].shape[0])
    c3 = int(state_dict["features.6.weight"].shape[0])
    hidden_dim = int(state_dict["classifier.1.weight"].shape[0])
    flat_dim = int(state_dict["classifier.1.weight"].shape[1])
    n_classes = int(state_dict["classifier.4.weight"].shape[0])

    pool_area = flat_dim // c3
    pool_hw = int(round(math.sqrt(pool_area)))
    if pool_hw * pool_hw != pool_area:
        raise ValueError(f"Unexpected digit checkpoint pool area: {pool_area}")

    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "pool_hw": pool_hw,
        "hidden_dim": hidden_dim,
        "n_classes": n_classes,
    }


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


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


def warp_from_corners(raw_bgr: np.ndarray, corners_xy: np.ndarray, warp_size: int) -> np.ndarray:
    src = order_points(corners_xy).astype(np.float32)
    dst = np.array(
        [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(raw_bgr, M, (warp_size, warp_size))


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


def unletterbox_points(points_xy: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] - meta["pad_x"]) / meta["scale"]
    pts[:, 1] = (pts[:, 1] - meta["pad_y"]) / meta["scale"]
    pts[:, 0] = np.clip(pts[:, 0], 0, meta["orig_w"] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, meta["orig_h"] - 1)
    return pts


def corners_from_segmentation_prob(prob: np.ndarray, post_thr: float = 0.5):
    mask = (prob >= post_thr).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Segmentation produced no foreground contour.")

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 10:
        raise RuntimeError("Segmentation contour area too small.")

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        pts = approx.reshape(4, 2).astype(np.float32)
        source = "approx_quad"
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)
        source = "min_area_rect"

    pts = order_points(pts)

    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.polylines(debug, [pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(debug, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(debug, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    meta = {
        "contour_area": float(area),
        "corner_source": source,
        "post_threshold": float(post_thr),
    }
    return pts, debug, meta

def extract_equal_cells(warped_bgr: np.ndarray, trim_frac: float = 0.12):
    h, w = warped_bgr.shape[:2]
    xs = [int(round(v)) for v in np.linspace(0, w - 1, 10)]
    ys = [int(round(v)) for v in np.linspace(0, h - 1, 10)]
    xs[0], xs[-1] = 0, w - 1
    ys[0], ys[-1] = 0, h - 1

    cells, boxes = [], []
    for r in range(9):
        for c in range(9):
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            cell_w = max(1, x1 - x0)
            cell_h = max(1, y1 - y0)
            tx = int(round(cell_w * trim_frac))
            ty = int(round(cell_h * trim_frac))
            nx0 = min(max(0, x0 + tx), w - 1)
            ny0 = min(max(0, y0 + ty), h - 1)
            nx1 = max(nx0 + 1, min(w, x1 - tx))
            ny1 = max(ny0 + 1, min(h, y1 - ty))
            cells.append(warped_bgr[ny0:ny1, nx0:nx1])
            boxes.append((nx0, ny0, nx1, ny1))
    return cells, boxes


def safe_standardize(feat: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(model["std"], dtype=np.float64).reshape(-1)
    std_safe = np.maximum(std, 1e-6)
    x = (feat.astype(np.float64) - mean) / std_safe
    x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
    x = np.clip(x, -20.0, 20.0)
    return x.astype(np.float32)


def preprocess_occ_gray(crop_bgr: np.ndarray, image_size: int) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return gray


def occ_feature_from_gray(gray: np.ndarray) -> np.ndarray:
    size = gray.shape[0]
    flat = gray.reshape(-1)
    ink_ratio = float((gray < 0.75).mean())
    center = gray[size // 4: 3 * size // 4, size // 4: 3 * size // 4]
    center_ink_ratio = float((center < 0.75).mean())
    row_sums = gray.mean(axis=1)
    col_sums = gray.mean(axis=0)
    feat = np.concatenate(
        [
            flat,
            np.array([ink_ratio, center_ink_ratio], dtype=np.float32),
            row_sums.astype(np.float32),
            col_sums.astype(np.float32),
        ]
    ).astype(np.float32)
    return feat


def baseline_occ_logit_and_prob(crop_bgr: np.ndarray, model: dict[str, Any]) -> tuple[float, float]:
    gray = preprocess_occ_gray(crop_bgr, model["image_size"])
    feat = occ_feature_from_gray(gray)
    feat_s = safe_standardize(feat, model)
    w = np.asarray(model["w"], dtype=np.float64).reshape(-1)
    b = float(np.asarray(model["b"]).reshape(()))
    z = float(np.dot(feat_s.astype(np.float64), w) + b)
    if not np.isfinite(z):
        z = 0.0
    z = float(np.clip(z, -30.0, 30.0))
    p = float(sigmoid_np(np.array([z], dtype=np.float64))[0])
    return z, p


@torch.no_grad()
def digit_logits_for_crops(crops: list[np.ndarray], model: nn.Module, image_size: int, device: str) -> np.ndarray:
    xs = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        xs.append(gray[None, :, :])
    x = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    logits = model(x).detach().cpu().numpy().astype(np.float32)
    return logits


def apply_occ_calibration(logits: np.ndarray, cal: dict[str, Any] | None) -> np.ndarray:
    z = logits.astype(np.float32)
    if cal is None or cal.get("kind") == "identity":
        out = z
    elif cal["kind"] == "platt":
        out = cal["a"] * z + cal["b"]
    else:
        raise ValueError(cal)
    return sigmoid_np(out)


def apply_digit_calibration(logits: np.ndarray, cal: dict[str, Any] | None) -> np.ndarray:
    z = logits.astype(np.float32)
    if cal is None or cal.get("kind") == "identity":
        zz = z
    elif cal["kind"] == "temp":
        zz = z / max(float(cal["t"]), 1e-3)
    elif cal["kind"] == "vector":
        zz = z * cal["a"][None, :] + cal["b"][None, :]
    else:
        raise ValueError(cal)
    return softmax_np(zz, axis=1)


def greedy_grid_from_calibrated(occ_probs: np.ndarray, digit_probs: np.ndarray, occ_threshold: float = 0.5) -> np.ndarray:
    pred = np.zeros(81, dtype=int)
    for i in range(81):
        if float(occ_probs[i]) >= occ_threshold:
            pred[i] = int(np.argmax(digit_probs[i])) + 1
    return pred.reshape(9, 9)


def load_occupancy_model(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "w": data["w"].astype(np.float32),
        "b": float(data["b"][0]),
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
        "image_size": int(data["image_size"][0]),
    }


def load_digit_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    image_size = int(ckpt["image_size"])
    state_dict = ckpt["model_state"]

    inferred = infer_digit_cnn_kwargs_from_state_dict(state_dict)
    model = SmallDigitCNN(
        image_size=image_size,
        n_classes=inferred["n_classes"],
        c1=inferred["c1"],
        c2=inferred["c2"],
        c3=inferred["c3"],
        pool_hw=inferred["pool_hw"],
        hidden_dim=inferred["hidden_dim"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, image_size, inferred


@lru_cache(maxsize=1)
def load_runtime() -> dict[str, Any]:
    frozen = load_frozen_paths()

    seg_ckpt = torch.load(frozen["segmentation"], map_location="cpu")
    seg_model = BoardUNet(in_channels=3, base_channels=int(seg_ckpt["base_channels"])).to(DEVICE)
    seg_model.load_state_dict(seg_ckpt["model_state"])
    seg_model.eval()

    occ_model = load_occupancy_model(frozen["occupancy"])
    digit_model, digit_image_size, digit_info = load_digit_model(frozen["digit"], DEVICE)

    with open(frozen["occ_calibration"], "r", encoding="utf-8") as f:
        occ_cal = json.load(f)
    with open(frozen["digit_calibration"], "r", encoding="utf-8") as f:
        digit_cal = json.load(f)

    return {
        "frozen": frozen,
        "seg_model": seg_model,
        "seg_image_size": int(seg_ckpt["image_size"]),
        "occ_model": occ_model,
        "digit_model": digit_model,
        "digit_image_size": digit_image_size,
        "digit_info": digit_info,
        "occ_cal": occ_cal,
        "digit_cal": digit_cal,
    }


def predict_mask_prob_letterbox(seg_model: nn.Module, raw_bgr: np.ndarray, image_size: int, device: str):
    boxed, lb_meta = letterbox_bgr(raw_bgr, image_size)
    img = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(img, (2, 0, 1))[None, :, :, :]).to(device)
    with torch.no_grad():
        logits = seg_model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32), lb_meta, boxed


def infer_board_outputs_from_crops(crops: list[np.ndarray], runtime: dict[str, Any]) -> dict[str, Any]:
    occ_logits = []
    occ_probs = []
    for crop in crops:
        z, p = baseline_occ_logit_and_prob(crop, runtime["occ_model"])
        occ_logits.append(z)
        occ_probs.append(p)
    occ_logits = np.asarray(occ_logits, dtype=np.float32)
    occ_probs = np.asarray(occ_probs, dtype=np.float32)
    digit_logits = digit_logits_for_crops(crops, runtime["digit_model"], runtime["digit_image_size"], DEVICE)
    return {
        "occ_logits": occ_logits,
        "occ_probs": occ_probs,
        "digit_logits": digit_logits,
    }


def predict_givens_from_bgr(raw_bgr: np.ndarray) -> list[list[int]]:
    runtime = load_runtime()
    frozen = runtime["frozen"]

    prob, lb_meta, _ = predict_mask_prob_letterbox(runtime["seg_model"], raw_bgr, runtime["seg_image_size"], DEVICE)
    pred_pts_lb, _, _ = corners_from_segmentation_prob(prob, post_thr=0.5)
    pred_pts_orig = unletterbox_points(pred_pts_lb, lb_meta)

    warp = warp_from_corners(raw_bgr, pred_pts_orig, warp_size=int(frozen["warp_size"]))
    crops, _ = extract_equal_cells(warp, trim_frac=float(frozen["trim_frac"]))
    outputs = infer_board_outputs_from_crops(crops, runtime)

    occ_probs = apply_occ_calibration(np.asarray(outputs["occ_logits"], dtype=np.float32), runtime["occ_cal"])
    digit_probs = apply_digit_calibration(np.asarray(outputs["digit_logits"], dtype=np.float32), runtime["digit_cal"])

    pred_grid = greedy_grid_from_calibrated(
        occ_probs,
        digit_probs,
        occ_threshold=float(frozen["occ_threshold"]),
    )
    return pred_grid.tolist()


def predict_givens_from_image(image_path: Path) -> list[list[int]]:
    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return predict_givens_from_bgr(raw_bgr)
