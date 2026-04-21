# MODEL_PROVENANCE.md

## Purpose

This file documents where each frozen V1 model artifact came from, why it is the blessed artifact, and what still remains to be proven for full retraining reproducibility.

This is a provenance document, not a benchmark report.

---

## Frozen V1 stack

The frozen V1 image-solver stack is:

- letterbox-trained segmentation
- original-image OCR warp
- equal-split crops
- current occupancy baseline
- current Chars74K transfer digit CNN
- final calibrated no-decode readout:
  - `occ_platt_digit_temp_no_decode`

---

## 1. Segmentation

### Blessed artifact
- `models/frozen_v1/segmentation/letterbox_seg_checkpoint.pt`

### Source
- **NB15**
- notebook: `15_stagewise_combined_inline_eval.ipynb`

### Why this is the blessed artifact
NB15 is the notebook where the controlled segmentation retraining comparison was run inside the main evaluation notebook.
That notebook trained both:
- stretch-trained segmentation
- letterbox-trained segmentation

The later roadmap snapshot records that:
- letterbox beat stretch on the downstream end-to-end metric that matters
- letterbox was then locked as the V1 production geometry baseline

### Promotion evidence
- letterbox default warp exact givens: **85.12%**
- stretch default warp exact givens: **82.64%**
- letterbox mean corner MAE: **4.65 px**
- stretch mean corner MAE: **6.07 px**

### Current provenance confidence
- **High confidence** that NB15 is the correct provenance source
- **Medium confidence** on full retraining reproducibility until the exact NB15 training/save config is copied into `scripts/train_segmentation_v1.py`

### Remaining gap
Still need to confirm from NB15:
- exact image size
- base channels
- epochs
- learning rate
- batch size
- seed
- exact save cell / save path details

---

## 2. Occupancy

### Blessed artifact
- `models/frozen_v1/occupancy/occupancy_model.npz`

### Source
- script: `scripts/train_occupancy_baseline_export_artifact.py`

### Why this is the blessed artifact
This script explicitly exports the saved occupancy artifact used by the end-to-end reader.
This is the cleanest provenance chain among the frozen base models.

### Current provenance confidence
- **High confidence**

### Remaining gap
Need to record in a final training manifest:
- blacklist path used for the cleaned baseline
- exact command line / config used for the exported artifact
- whether the shipped artifact was created before or after any later notebook-only experiments

---

## 3. Digit recognizer

### Blessed artifact
- `models/frozen_v1/digits/digit_cnn.pt`

### Source
- script family: `scripts/train_digit_cnn_benchmarks.py`
- decision note: Day 21 CNN benchmark
- winning model family: **Chars74K transfer CNN**

### Why this is the blessed artifact
The Day 21 benchmark decision records:
- Chars74K transfer = best validation result
- additional pretraining-source churn was not worth continuing
- Chars74K transfer became the frozen digit recognizer choice

### Current provenance confidence
- **Medium / high confidence**

### Remaining gap
Need to record in a final training manifest:
- exact benchmark command/config that produced the shipped `digit_cnn.pt`
- image size
- epochs
- pretrain epochs
- batch size
- learning rate
- weight decay
- seed

---

## 4. Final readout / calibration layer

### Blessed exported files
- `models/frozen_v1/calibration/occ_calibration.json`
- `models/frozen_v1/calibration/digit_calibration.json`
- `models/frozen_v1/calibration/calibration_manifest.json`

### Source
- **NB20**
- notebook: `20_day28_calibration_constrained_decode_v5.ipynb`

### Why this is the blessed final readout
The final shipped V1 behavior was not saved directly from notebook memory during NB20, because:
- `SAVE_NOTEBOOK_ARTIFACTS=False`

So the notebook itself is the source of truth for:
- final variant label
- occupancy threshold
- occupancy calibration params
- digit calibration params
- final combined metric
- final latency

These values were then exported into JSON files by:
- `scripts/freeze_calibration_v1.py`

### Frozen final readout
- variant: `occ_platt_digit_temp_no_decode`
- occupancy threshold: `0.35`

### Frozen occupancy calibration
- kind: `platt`
- `a = 0.7386137843132019`
- `b = -0.34071943163871765`

### Frozen digit calibration
- kind: `temp`
- `t = 0.6822174787521362`

### Frozen performance line
Combined non-Kaggle (`core_val + core_test`):
- exact givens: **0.859504**

Held-out non-Kaggle `core_test`:
- exact givens: **0.887097**

Hot full-path latency:
- mean: **233.206886 ms**
- p95: **239.557425 ms**

### Current provenance confidence
- **High confidence**

### Remaining gap
None on export provenance.
The remaining task is only to keep the exported JSON files and manifest in sync.

---

## 5. What is fully frozen now

The following are now frozen in repo-tracked form:

- base segmentation artifact
- base occupancy artifact
- base digit artifact
- final calibration files
- frozen manifest
- final shipped readout label
- final threshold
- final frozen metric line
- final frozen latency line

---

## 6. What is not fully closed yet

### A. Segmentation retrain reproducibility
Need to verify that `scripts/train_segmentation_v1.py` exactly matches the final NB15 training config.

### B. Occupancy and digit training manifests
Need machine-readable manifests that record:
- source script
- split policy
- blacklist path if applicable
- hyperparameters
- seed
- output artifact path

### C. Regression protection
Still needed:
- small non-Kaggle gold set
- gold-set test
- official metric regression test

---

## 7. Decision summary

### Confidence by component
- segmentation artifact identity: **high**
- segmentation retrain reproducibility: **not fully proven yet**
- occupancy artifact identity: **high**
- digit artifact identity: **medium/high**
- final calibration/readout provenance: **high**

### Bottom line
The V1 frozen inference stack is now well identified.

The remaining MLOps work is no longer about “which files are the real model.”
It is now about:
- retraining reproducibility
- regression protection
