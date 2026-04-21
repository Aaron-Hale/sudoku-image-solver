# FREEZE_CONTRACT.md

## Purpose

This document freezes the exact V1 Sudoku image-solver behavior so the GitHub packaging effort does not accidentally degrade performance.

This contract is the source of truth for:
- the blessed inference stack
- the official dataset/eval scope
- the official reported metrics
- the exact artifacts/configs that must not drift
- the regression checks the packaged repo must continue to pass

If the packaged repo changes behavior relative to this contract, the change is a regression unless explicitly approved and re-baselined.

---

## 1. Repo boundary

This repo begins from:

**labeled data / training-ready artifacts onward**

In scope:
- model training
- model evaluation
- calibration
- image-to-givens inference
- solver integration
- reproducible metrics
- debug/demo outputs

Out of scope:
- raw image labeling workflow
- annotation tooling
- manual corner-label creation process
- full dataset archaeology / labeling history

Dataset structure and split naming follow the existing project conventions:
- `data/raw/core_train`
- `data/raw/core_val`
- `data/raw/core_test`
- `data/labels/boards`
- `data/labels/cells`
- image IDs:
  - `ct_####`
  - `cv_####`
  - `cte_####`

---

## 2. Blessed V1 inference stack

### 2.1 Geometry front end
- **letterbox-trained segmentation** is the default geometry front end
- classical CV detector is historical/comparison only
- stretch-trained segmentation is comparison only
- adaptive warp is not the default production path

### 2.2 Board warp
- infer board corners with segmentation
- map predicted corners back to **original image coordinates**
- perform the final OCR warp from the **original-resolution image**
- do **not** use the resized segmentation image for the final OCR warp

### 2.3 OCR crop method
- use **equal-split** cell crops as the blessed OCR crop method
- refit/grid-box logic is retained for parser-side or historical use, but is not the blessed public V1 OCR crop path

### 2.4 Occupancy model
- use the cleaned Day 18 occupancy baseline artifact
- artifact family: saved exported inference-ready occupancy artifact
- do **not** promote occupancy CNN variants in the packaged V1 path

### 2.5 Digit model
- use the **Chars74K transfer CNN** as the blessed digit recognizer
- do **not** promote weaker linear baseline or abandoned alternates into the packaged V1 path

### 2.6 Final readout
- use the calibrated no-decode variant:
  - **`occ_platt_digit_temp_no_decode`**
- no default false-empty override
- no default CLAHE / sharpen
- no default conditional large warp
- no aggressive constrained decoding in the main path

Reason for final readout choice:
- same success metrics as the vector no-decode variant
- slightly lower latency
- simpler production choice

---

## 3. Blessed inference artifacts

Freeze the exact deployable inference artifacts the packaged repo will load.

### 3.1 Segmentation inference artifact
- `models/frozen_v1/segmentation/letterbox_seg_checkpoint.pt`

Must contain all metadata needed to rebuild the exact inference model, including:
- `model_state`
- `image_size`
- `base_channels` or equivalent architecture metadata

### 3.2 Occupancy inference artifact
- `models/frozen_v1/occupancy/occupancy_model.npz`

Must contain the exact exported inference-ready occupancy artifact used by the packaged repo.

### 3.3 Digit inference artifact
- `models/frozen_v1/digits/digit_cnn.pt`

Must contain all metadata needed to rebuild the exact inference model, including:
- `model_state`
- `image_size`
- class mapping / label metadata if needed

### 3.4 Calibration artifacts
The frozen final readout is backed by:
- `models/frozen_v1/calibration/occ_calibration.json`
- `models/frozen_v1/calibration/digit_calibration.json`
- `models/frozen_v1/calibration/calibration_manifest.json`

### 3.5 No silent artifact swaps
Any regression run must print:
- segmentation artifact path
- occupancy artifact path
- digit artifact path
- calibration artifact identifiers

so accidental artifact drift is obvious.

### 3.6 Provenance anchors
The official provenance anchors for the frozen V1 stack are:

- **Segmentation**
  - source notebook: `15_stagewise_combined_inline_eval.ipynb`
  - reproducibility script: `scripts/train_segmentation_v1.py`

- **Occupancy**
  - source script: `scripts/train_occupancy_baseline_export_artifact.py`

- **Digit**
  - source script family: `scripts/train_digit_cnn_benchmarks.py`

- **Final readout / calibration**
  - source notebook: `20_day28_calibration_constrained_decode_v5.ipynb`
  - export script: `scripts/freeze_calibration_v1.py`

Note:
- NB20 was the canonical source of the final calibrated readout values because the final calibration state lived in notebook memory and was exported afterward into repo-tracked JSON files.

---

## 4. Blessed config values

These values must be frozen explicitly in code and docs.

### Core image-solver settings
- `DEFAULT_WARP_SIZE = 900`
- `TRIM_FRAC = 0.12`
- final no-decode variant:
  - `occ_platt_digit_temp_no_decode`
- final occupancy threshold:
  - `0.35`
- no-decode path:
  - `True`

### Final frozen calibration values

#### Occupancy calibration
- kind: `platt`
- `a = 0.7386137843132019`
- `b = -0.34071943163871765`

#### Digit calibration
- kind: `temp`
- `t = 0.6822174787521362`

### Data inclusion policy
- packaged training/eval code may support Kaggle-inclusive training
- the **official reported metric** is still:
  - **non-Kaggle evaluation only**
- Kaggle-tagged boards are excluded from the official reported eval slice

### Device/latency note
- hot steady-state latency is measured after model load / warmup
- it is not the same as cold-start or full user-perceived UI latency

---

## 5. Official evaluation contract

### 5.1 Main metric
**Exact givens match**

Definition:
- compare prediction vs ground truth on **true given cells only**
- a board passes if **every true given cell** is correct
- false positives on truly empty cells do **not** directly change exact-givens status, but are tracked separately in other metrics

### 5.2 Supporting metrics
Track and report:
- exact givens match rate
- mean givens accuracy
- mean full-board cell accuracy
- `missed_as_empty`
- `wrong_digit`
- false positive on empty
- legality failure rate
- latency

### 5.3 Official reported slices

#### Held-out reference point
- non-Kaggle `core_test`
- trusted best reference from the current project history:
  - exact givens match: **0.887097**
  - `wrong_digit`: **5**
  - legality failure rate: **4.84%**

#### Combined non-Kaggle development summary
- combined `core_val + core_test`
- Kaggle-tagged boards excluded
- trusted earlier combined non-Kaggle reference:
  - evaluated boards: **121**
  - exact givens match: **84.30%**
  - mean givens accuracy: **97.09%**
  - `missed_as_empty`: **83**
  - `wrong_digit`: **19**

#### Frozen calibrated no-decode combined summary
- combined `core_val + core_test`
- Kaggle-tagged boards excluded
- frozen best recent calibrated no-decode path:
  - exact givens: **0.859504**
  - mean givens accuracy: **0.975222**
  - mean full-board cell accuracy: **0.988369**
  - total `missed_as_empty`: **71**
  - total `wrong_digit`: **22**
  - total false positive on empty: **21**
  - legality failure rate: **0.082645**

### 5.4 Latency contract
Current hot steady-state image-read-to-givens reference:
- mean: **233.206886 ms**
- p95: **239.557425 ms**

This should be described as:
- hot steady-state
- models already loaded
- warmed path
- image-read-to-givens inference
not cold-start end-user latency

---

## 6. Known-good interpretation of the current system

### What is solved enough for V1
- geometry is solved enough for V1
- letterbox segmentation is the correct production front end
- original-image warp is the correct OCR warp path
- the current digit recognizer is good enough
- the current occupancy baseline is good enough

### What remains imperfect
- remaining misses are concentrated in a minority of hard boards
- dominant remaining failure mode is still **filled cells dropped as empty**
- hard cases cluster around:
  - small boards in the original image
  - skew / tilt
  - blur / faint digits
  - post-geometry quality loss
  - a few real digit ambiguities such as `6 vs 8`

### What is explicitly not part of the blessed V1 path
- occupancy CNN promotion
- broad decode search
- CLAHE-first production path
- conditional large-warp default
- any approach that only looked good on val but regressed on held-out behavior

---

## 7. Regression harness requirements

Before any refactor / repo cleanup is accepted, the packaged repo must pass:

### 7.1 Gold-set inference test
A fixed gold set of `N=10–20` images must be stored with expected outputs:
- expected givens grid
- optional expected solved grid
- optional expected warning/confidence flags

The packaged repo must reproduce the expected givens outputs exactly on the gold set.

### 7.2 Official metric regression
A regression script must reproduce the official metric slice(s) using the frozen artifacts:
- held-out non-Kaggle `core_test`
- optionally combined non-Kaggle `core_val + core_test`

The result must stay within a tiny tolerance of the frozen metrics.

### 7.3 Latency sanity check
A hot-path latency benchmark must confirm the packaged repo remains in the same general range as the frozen implementation.

### 7.4 Artifact identity check
The regression harness must print artifact identifiers and config values before evaluation:
- segmentation artifact
- occupancy artifact
- digit artifact
- calibration artifact/config
- warp size
- trim fraction
- occ threshold
- final readout variant

---

## 8. Packaging rules

### 8.1 Copy, do not recreate
The first packaging pass must:
- extract the exact trusted inference path
- wrap it into repo modules
- avoid redesigning or simplifying behavior prematurely

### 8.2 Keep the public path narrow
The clean public repo should primarily expose:
- inference entrypoint
- evaluation entrypoint
- training entrypoints for the chosen models
- solver integration
- examples/demo assets
- README + metrics + alternatives tested

### 8.3 Archive experiments separately
Old experiments may be preserved, but must not be part of the default public path:
- failed occupancy variants
- CLAHE experiments
- decode variants that did not win
- abandoned notebook branches

### 8.4 Repo sequencing
Build the clean Sudoku image-solver repo **before** moving into AR.
AR should be treated as a follow-on system built on top of the frozen perception stack, not as the next place to keep debugging the model.

---

## 9. Remaining open items before the contract is truly final

The major artifact/config TODOs are now resolved.

What still remains open is:
- gold-set image IDs for the regression suite
- proof that `scripts/train_segmentation_v1.py` exactly matches the final NB15 training config
- finalized machine-readable training manifests for segmentation / occupancy / digit if desired

Until the gold-set regression set is frozen, this document should still be treated as **operationally close to final**, but not yet fully complete.

---

## 10. Minimal acceptance summary

The packaged repo is acceptable only if it continues to satisfy all of the following:

- blessed stack:
  - letterbox segmentation
  - original-image OCR warp
  - equal-split crops
  - current occupancy baseline
  - current digit CNN
  - `occ_platt_digit_temp_no_decode`
- official non-Kaggle metric behavior remains consistent
- held-out non-Kaggle `core_test` still reproduces the trusted reference envelope
- hot steady-state latency remains in the same practical range
- no silent artifact/config drift occurs
- raw labeling workflow remains out of scope for the public repo