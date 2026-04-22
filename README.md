# Sudoku Image Solver

A reproducible Sudoku image-solver for **real camera photos and webcam frames** of **printed 9x9 Sudoku boards**.

> **Important:** These benchmarks should not be compared as apples-to-apples.
> Different systems were trained and evaluated on different datasets, with different image quality, framing, board size, distortion, and reporting rules.
> This repo’s evaluation slice includes harder real-photo cases such as **small puzzles in frame**, **skew / tilt**, **blur / faint digits**, and other post-geometry OCR difficulties, so direct comparison to cleaner close-up datasets can be misleading.

---

## Headline metrics

All public metrics below are shown on the combined **non-Kaggle `core_val + core_test`** slice.

| Metric | Value |
|---|---:|
| Cell accuracy | **98.84%** |
| Board accuracy | **85.95%** |

### Why two metrics?

- **Cell accuracy** is useful because it is common in papers and repos.
- **Board accuracy** is the metric that better matches end-user trust. One wrong clue can make an otherwise strong read untrustworthy.

In this repo:
- **Cell accuracy** = **mean full-board cell accuracy**
- **Board accuracy** = **exact givens match**

---

## Example predictions

Below are two real examples from the project showing the geometry step and the post-warp cell-level prediction overlay.

### Example 1 — `cv_0002`

**Pre-warp / geometry debug**
![cv_0002 geometry debug](docs/images/02_cv_0002_geometry_debug.jpg)

**Post-warp / prediction overlay**
![cv_0002 prediction overlay](docs/images/02_cv_0002_overlay.jpg)

### Example 2 — `cv_0003`

**Pre-warp / geometry debug**
![cv_0003 geometry debug](docs/images/03_cv_0003_geometry_debug.jpg)

**Post-warp / prediction overlay**
![cv_0003 prediction overlay](docs/images/03_cv_0003_overlay.jpg)

These examples help show why direct benchmark comparison is tricky: this repo was built and evaluated on real-photo cases that include small puzzles in frame, skew / tilt, blur, faint digits, and other OCR difficulties rather than only clean close-up crops.

## Benchmark context

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

This table is meant as **context**, not a strict leaderboard.

The most meaningful comparisons are:
- other systems that read Sudoku from **real images**
- other systems that report **board-level** accuracy, not only digit accuracy
- systems whose task is closer to **printed-camera-photo OCR**, not only synthetic Visual Sudoku

---

## Why board accuracy matters more than cell accuracy

Cell accuracy is common because it is easy to report and compare.

But for real user trust, **board accuracy** is usually the better metric:
- if one clue is wrong, the user may not trust the whole solve
- a system can have very high cell accuracy and still miss too many boards end-to-end
- this repo is a practical example of that gap

That is why this repo leads with:
- **Cell accuracy = 98.84%**
- **Board accuracy = 85.95%**

instead of pretending the cell number alone captures the user experience.

---

## Practical path to improve board accuracy on video

A promising next step for video feeds is **temporal ensembling across 3 adjacent frames**.

Instead of trusting one frame:
1. run the frozen model on 3 nearby frames of the same puzzle
2. vote per cell on **empty vs filled**
3. vote on the digit label, or sum calibrated probabilities
4. output the ensembled board

Why this is attractive:
- the remaining errors are concentrated in a small number of hard cells
- adjacent frames often provide slightly different views of faint or ambiguous digits
- this can improve **board accuracy** without changing the frozen single-frame model

This is a future enhancement idea, not a number already claimed in the benchmark table.

---

## Frozen V1 pipeline

The frozen production path is:

- **letterbox-trained segmentation** for board localization
- predicted corners mapped back to **original image coordinates**
- final OCR warp from the **original-resolution image**
- **equal-split** 9x9 cell crops
- cleaned **occupancy baseline** artifact
- **Chars74K transfer CNN** for digits
- calibrated no-decode readout:
  - `occ_platt_digit_temp_no_decode`

Key frozen config:
- `DEFAULT_WARP_SIZE = 900`
- `TRIM_FRAC = 0.12`
- `occ_threshold = 0.35`

---

## Approaches tested

This repo keeps the final winner, but the project got there by testing several alternatives.

### Geometry
- classical CV outer-board detection
- segmentation as the geometry upgrade
- stretch-trained segmentation
- letterbox-trained segmentation
- original-image OCR warp vs resized-image OCR warp

### OCR crops
- equal-split crops
- refit / grid-box alternatives

### Occupancy
- initial baseline
- blacklist cleanup
- later occupancy-focused experiments that did not win promotion

### Digit recognition
- linear softmax baseline
- cells-only CNN
- MNIST transfer
- EMNIST transfer
- Chars74K only
- Chars74K transfer

### Final readout
- baseline no-calibration path
- calibrated no-decode variants
- more aggressive constrained-decoding variants that did not win promotion

---

## Major decisions that stuck

- **Segmentation** replaced the classical detector as the main front end.
- **Letterbox-trained segmentation** beat stretch-trained segmentation on the downstream metric that mattered.
- The final OCR warp should come from the **original image**, not the resized segmentation image.
- **Equal split** remained the best default crop method for the public V1 path.
- **Chars74K transfer CNN** became the digit recognizer.
- The final shipped readout was:
  - `occ_platt_digit_temp_no_decode`

---

## Repo layout

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

