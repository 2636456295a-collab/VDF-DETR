# Depth-input intervention protocol

Corresponding to Section 3.4 and Table 3 of the manuscript.

## Purpose

Quantify the contribution of **valid depth readings** to surfacing-fish detection, separated from the RGB appearance branch.

## Conditions

| Condition | Depth input | Validity mask M |
|---|---|---|
| Normal | original aligned depth D | original M |
| Depth removed | zeroed / invalid depth | all-invalid M |
| Depth mismatched | depth maps shuffled across images | original M |

All three conditions use the **same trained VDF-DETR weights and the same images**; only the depth input is altered at inference time. No retraining is performed, so differences are attributable to the depth input itself.

## Data

525 validation images / 2 550 `surfacing` boxes. This is a **different split** from the 525-image test set used for Tables 1–2. Keep the two partitions separate when reproducing.

## Metrics

P, R, F1, AP50 and AP50–95 computed with the same unified `score()` and COCO evaluation path used for the main comparison (`tools/formal50_unified_eval.py`). Per-model F1-optimal confidence thresholds are reused, and matching IoU is 0.50.

## Reported outcome

The normal-depth condition raises recall by **4.63 percentage points** over the depth-removed condition, while the mismatched condition falls back. This supports the interpretation that VGDF uses valid depth to recover missed detections rather than merely responding to an additional input channel.
