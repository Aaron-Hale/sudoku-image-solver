# Sudoku Image Solver

A frozen V1 system for reading **printed 9x9 Sudoku boards from real camera photos and webcam frames**.

Unlike synthetic-board or perfectly cropped close-up pipelines, this project targets real-photo OCR failure modes such as **small puzzles in frame, skew / tilt, blur, faint digits, and post-geometry quality loss**.

Using **440 training images** and **121 held-out evaluation boards**, the frozen system achieves:
- **85.95% board accuracy** (exact givens match)
- **98.84% cell accuracy**
- **233.2 ms mean** hot steady-state latency (**239.6 ms p95**)

This public repo starts from **labeled data / training-ready artifacts onward** and documents the frozen inference path, evaluation contract, artifact provenance, and final engineering decisions.

## Why lead with board accuracy?

Cell-level accuracy is useful, but it overstates end-user quality for Sudoku OCR.

If even one given is wrong, the board may be unusable. That makes **exact givens match** the more meaningful top-line metric for a practical Sudoku image solver.

In this repo:
- **Board accuracy** = exact match on all true given cells for a board
- **Cell accuracy** = mean full-board cell accuracy across all 81 cells

---

## What this repo is

This repo starts from **labeled data / training-ready artifacts onward**.

### In scope
- model training
- model evaluation
- calibration
- image-to-givens inference
- solver integration
- reproducible metrics
- debug and demo outputs

### Out of scope
- raw image-labeling workflow
- annotation tooling
- manual corner-label creation process
- full dataset archaeology / labeling history

This is a **clean public artifact** for the frozen system, not the full private project history.

---

## Problem setting

The target problem is not synthetic Sudoku and not only perfectly cropped close-up boards.

The evaluation includes real-photo OCR difficulties such as:
- small puzzles in frame
- skew / tilt
- blur
- faint digits
- post-geometry quality loss

That makes this a more practical printed-camera-photo OCR task, but it also makes direct comparison to cleaner or synthetic benchmarks misleading.

---

## Frozen V1 pipeline

The frozen production path is:

1. **Letterbox-trained segmentation** for board localization
2. Predicted corners mapped back to **original image coordinates**
3. Final OCR warp from the **original-resolution image**
4. **Equal-split** 9x9 cell crops
5. Cleaned exported **occupancy baseline** artifact
6. **Chars74K transfer CNN** for digit recognition
7. Calibrated no-decode readout:
   - `occ_platt_digit_temp_no_decode`

### Frozen config
- `DEFAULT_WARP_SIZE = 900`
- `TRIM_FRAC = 0.12`
- `occ_threshold = 0.35`

### Frozen calibration
- Occupancy calibration: Platt scaling
- Digit calibration: temperature scaling

---

## Stagewise summary

### 1) Geometry: finding the board
The first problem was simply finding the Sudoku board reliably in a full image. The project started with a **classical CV detector** based on contours and quadrilateral selection. That worked well for close-up boards, but it broke when the puzzle was small in frame, blurred, cluttered, or competing with other rectangular structures.

To fix that, the project moved to a **trained segmentation model** using the existing board-corner labels already available in the dataset. That solved most of the geometry problem immediately. On `core_val`, the classical front end reached only **18.64%** exact board match, while segmentation reached **76.27%**, close to **77.97%** with oracle label corners.

The final geometry refinement was choosing **how to preprocess images for segmentation**. A controlled comparison showed that **letterbox-trained segmentation** (aspect-ratio-preserving resize with padding) was better than stretch-trained segmentation on the downstream metric that mattered most. It got **97.52%** of boards with all 4 corners within **25 px**, versus **93.39%** for stretch, and also produced better end-to-end exact givens match (**85.12%** vs **82.64%**). That is why letterbox became the frozen V1 geometry front end.

### 2) Turning the board into OCR-ready cells
In parallel, the project also evaluated how a warped board should be converted into stable **cell crops** for OCR. The two main options were simple **equal-split crops** versus **refit/grid-box crops** that try to follow the inner lattice more closely.

Refit was appealing because it could help when the warp was imperfect, but it also added complexity and was not obviously better on most boards. A manual A/B review on **40 validation boards** showed that equal split was already strong enough: **11 boards favored equal split, 3 favored refit, and 26 were ties**. Refit remained useful for parser-side debugging and imperfect-warp analysis, but it was not a large enough win to justify making it the default public inference path.

That is why the frozen V1 system uses **equal-split crops** as the default OCR path.

### 3) OCR: separating occupancy from digit recognition
The OCR problem was split into two stages:
1. **Occupancy**: is a cell empty or filled?
2. **Digit recognition**: if filled, which digit is it?

That split mattered because those two tasks fail differently. Occupancy is easier in principle, while digit recognition is a harder shape-classification problem. After blacklist cleanup, the occupancy model was already strong enough to keep as the main stage, reaching **98.47%** validation accuracy with **99.01%** precision on filled cells and **97.14%** recall on filled cells.

For digits, the project first tried a **linear softmax baseline**, which reached only **73.8%** validation accuracy and clearly underfit the printed-digit task. The next step was to move to a **CNN**, which was much better aligned with the visual nature of the problem. In the Day 21 benchmark, the best setup was **Chars74K transfer** at **94.53%** validation accuracy, ahead of **cells only (93.63%)**, **MNIST transfer (94.26%)**, and **EMNIST transfer (94.37%)**.

That result made the final digit-model choice clear: use a **Chars74K-transfer CNN** as the production recognizer.

### 4) Stagewise performance vs end-to-end performance
One useful question was whether the system still had a basic OCR problem, or whether the remaining failures were caused by end-to-end compounding errors.

To answer that, the project measured OCR under **correct geometry** by using labeled corners. Under that setup, the OCR stack was already very strong:
- **99.72%** occupancy accuracy
- **99.59%** occupied-cell digit accuracy

But with full pure-model inference, the system still reached only:
- **84.30%** exact givens match
- **97.09%** mean givens accuracy

That gap showed the remaining problem was **not** basic OCR capacity and **not** a lack of heavier Sudoku logic. The main issue was end-to-end robustness on hard real-photo boards. More specifically, the dominant remaining failure mode was **filled cells being dropped as empty**: the combined held-out evaluation logged **83 `missed_as_empty` errors** versus only **19 `wrong_digit` errors**. Keeping occupancy separate from digit recognition made that bottleneck visible.

### 5) Final V1 direction
By this point, the project had enough evidence to stop treating the system as an open-ended experiment and lock a production path.

The frozen V1 system uses:
- **letterbox-trained segmentation** for geometry
- **original-image OCR warp** after board localization
- **equal-split crops** for OCR
- a separate **occupancy stage**
- a **Chars74K-transfer CNN** for digits
- a calibrated no-decode readout: `occ_platt_digit_temp_no_decode`

The final OCR-warp choice was also validated directly. Switching from the resized segmentation image to an **original-image OCR warp** improved downstream behavior while keeping latency roughly flat. On `core_val`, mean givens accuracy improved from **0.9519** to **0.9619**, `missed_as_empty` dropped from **68** to **52**, and `wrong_digit` dropped from **20** to **17**. On held-out `core_test`, exact givens match improved from **0.8710** to **0.8871**, and `wrong_digit` dropped from **10** to **5**.

A conservative false-empty override was tested as well, but it did **not** justify becoming the default path: it added major latency without meaningful held-out improvement. That is why the frozen V1 system locks in **letterbox segmentation + original-image OCR warp + equal-split crops + separate occupancy + Chars74K-transfer CNN + calibrated no-decode readout** as the final production direction.

---

## Major decisions that stuck

The final system was not the first baseline. The project tested multiple alternatives and froze the path that best balanced end-to-end accuracy, simplicity, and latency.

| Area | Frozen V1 choice | Alternatives not promoted to the public V1 path |
|---|---|---|
| Geometry front end | **Letterbox-trained segmentation** | Classical CV front end, stretch-trained segmentation, adaptive-warp default |
| OCR warp path | **Warp from original-resolution image** | Warp from resized segmentation image |
| OCR crop method | **Equal-split crops** | Refit / grid-box default path |
| Occupancy stage | **Cleaned exported baseline** | Later occupancy variants that did not clearly earn promotion |
| Digit recognizer | **Chars74K transfer CNN** | Linear softmax baseline, weaker abandoned variants |
| Final readout | **`occ_platt_digit_temp_no_decode`** | Default false-empty override, CLAHE-first path, aggressive constrained-decoding default |

### Why this readout stayed
The final readout matched the strongest practical success behavior while staying simpler and slightly lower-latency than the closest competing no-decode variant.

---

## Example predictions

Below is a sample from the project showing the geometry step and the post-warp cell-level prediction overlay.

### Raw Image — `cv_0003`

**Pre-warp / geometry debug**

![cv_0003 geometry debug](docs/images/03_cv_0003_geometry_debug.jpg)

**Post-warp / prediction overlay**

![cv_0003 prediction overlay](docs/images/03_cv_0003_overlay.jpg)

---

## Benchmark context

> [!IMPORTANT]
> **Do not read this as a strict leaderboard.**
> These systems were trained and evaluated on different datasets, with different image quality, framing, board size, distortion, and reporting rules. Several published approaches also used cleaner images with less noise and larger boards.
>
> This table is included to show the broader landscape and provide rough metric context, not to claim exact apples-to-apples ranking.
>
> This repo’s evaluation includes harder real-photo cases such as **small puzzles in frame**, **skew / tilt**, **blur / faint digits**, and other post-geometry OCR difficulties, so direct comparison to cleaner close-up datasets can be misleading.

| System | Cell accuracy | Board accuracy |
|---|---:|---:|
| **This repo** | **98.84%** | **85.95%** |
| **[Kainos Sudoku CV project](https://www.kainos.com/insights/blogs/ai-academy-capstone-projects--improving-document-data-extraction-through-contextualisation-computer-vision-based-sudoku-solver)** | — | **93.8%*** |
| **[PBCS / Sudoku Assistant (2024)](https://link.springer.com/article/10.1007/s10601-024-09372-9)** | **99.2%** | **94.84%** |
| **[Wicht / smartphone Sudoku dataset](https://github.com/wichtounet/sudoku_dataset)** | — | **87.5%**† |
| **[mineshpatel1/sudoku-solver](https://github.com/mineshpatel1/sudoku-solver)** | — | **99.2%** |
| **[Recurrent Transformer (ICLR 2023)](https://openreview.net/forum?id=udNhDCr2KQe)** | **99.77%** | **93.5%** |
| **[NeurASP](https://www.ijcai.org/proceedings/2020/0243.pdf)** | **96.9%** | **66.5%** |
| **[AS2 (2026)](https://arxiv.org/abs/2603.18436)** | **99.89%** | **100.0%**‡ |

\* Reported on “starting boards” only, which is closer to this repo’s intended use case than completed-board evaluation, but it is still a different dataset and protocol.  
† Wicht’s dataset page reports 12.5% error on one real-image setup, equivalent to 87.5% accuracy. This is a historical real-camera benchmark, not the same eval protocol as this repo.  
‡ AS2 reports 100% constraint satisfaction on Visual Sudoku, which is a synthetic / normalized benchmark and not directly comparable to a real printed-camera-photo OCR pipeline.

### How to read this table

The most meaningful comparisons are:
- other systems that read Sudoku from **real images**
- other systems that report **board-level** accuracy, not only digit accuracy
- systems whose task is closer to **printed-camera-photo OCR**, not only synthetic Visual Sudoku

---

## What is solved, and what is still hard

### Solved enough for V1
- geometry is solved enough for V1
- segmentation is the correct production front end
- original-image warp is the correct OCR warp path
- the current occupancy stage is good enough
- the current digit recognizer is good enough

### Remaining failure pattern
The remaining misses are concentrated in a minority of hard boards.

The dominant remaining failure mode is still:

**filled cells being dropped as empty**

Hard cases cluster around:
- small boards inside larger images
- skew / tilt
- blur
- faint digits
- post-geometry quality loss
- a few real ambiguities such as **6 vs 8**

### What this repo does not claim
- live AR-grade temporal stability
- fully solved scene-level board discovery
- a completely solved last-mile OCR problem

This is a strong frozen V1 system, not a claim that Sudoku image understanding is finished.

---

## Where the current V1 still breaks

Most of the remaining misses come from a small number of hard boards, not broad failure across the dataset. In practice, the residual errors fall into three buckets.

### 1) Residual warp / localization failures (**3 / 121 boards**)

These are usually smaller, farther-away boards where the puzzle occupies too few pixels before OCR. In these cases, the board is found, but the final warp is not clean enough to preserve all of the detail needed for reliable downstream recognition.

![Residual warp failure example](docs/images/failure_cases/core_test_cte_0043_miss_overlay.jpg)

### 2) Rare occupancy failure (**1 / 9,801 cells**)

Occupancy errors are now rare. Across **121 evaluation boards** (**9,801 total cells**), this issue appeared once. The representative failure below is included for completeness, but it is not a major driver of the remaining error.

![Rare occupancy failure example](docs/images/failure_cases/core_test_cte_0008_miss_overlay.jpg)

### 3) Wrong digit inference on small / low-quality puzzles (**13 / 121 boards**)

The largest remaining bucket is digit confusion on smaller, farther-away puzzles where the final OCR crop has limited detail. After segmentation, the system already performs the final OCR warp on the **original image resolution**, so the remaining mistakes are generally reasonable ambiguities such as **8 vs 6** or **9 vs 6**, not obvious model collapse.

A more aggressive correction layer that compares digits against Sudoku consistency constraints could reduce some of these errors, but it would add latency and complexity and is intentionally not part of the default V1 path.

![Wrong digit inference example](docs/images/failure_cases/core_test_cte_0014_miss_overlay.jpg)

### Practical takeaway

The remaining misses are concentrated in a narrow hard-case slice: **small / distant boards, reduced stroke quality, and a few visually ambiguous digits**. That is why the next likely gains come from better hard-case OCR handling, not from reworking the overall architecture.

## Reproducibility and evaluation

The repo is intentionally frozen around a narrow V1 path. A refactor should preserve behavior and should not silently swap artifacts or configs.

### Frozen artifact family
- `models/frozen_v1/segmentation/letterbox_seg_checkpoint.pt`
- `models/frozen_v1/occupancy/occupancy_model.npz`
- `models/frozen_v1/digits/digit_cnn.pt`
- `models/frozen_v1/calibration/occ_calibration.json`
- `models/frozen_v1/calibration/digit_calibration.json`
- `models/frozen_v1/calibration/calibration_manifest.json`

### Official evaluation policy
- Kaggle-tagged images are excluded from the public reported metrics
- the primary metric is **exact givens match**
- supporting metrics include mean givens accuracy, mean full-board cell accuracy, legality failure rate, and latency

### Running the frozen evaluation
The full frozen evaluation expects access to the original labeled data tree and raw images. Those assets are not bundled into this public repo.

```bash
export SUDOKU_DATA_ROOT=/path/to/private_sudoku_data
python scripts/run_frozen_eval_v1.py --splits core_val core_test --exclude-kaggle
pytest -q tests/test_metric_regression.py
```

---

## Repository layout

```text
models/frozen_v1/
  segmentation/
  occupancy/
  digits/
  calibration/
  manifest.json

src/sudoku_solver/
  frozen_config.py
  inference.py

scripts/
  train_segmentation_v1.py
  freeze_calibration_v1.py
  run_frozen_eval_v1.py

tests/
  goldset/
  test_goldset.py
  test_metric_regression.py

docs/
  MODEL_PROVENANCE.md
```

---

## Summary

This repo is meant to demonstrate three things:

1. A practical Sudoku OCR system for real photos, not just clean synthetic boards
2. A clear frozen production path with explicit artifact and metric discipline
3. Honest end-to-end evaluation where **board accuracy** is treated as the metric that matters most
