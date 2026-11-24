# Depth-input intervention protocol

Corresponding to Sections 2.5 and 3.3 and Table 5 of the manuscript.

## Purpose

Quantify the contribution of **valid depth readings** to surfacing-fish detection, separated from the RGB appearance branch.

## Conditions

| Condition | Depth input | Validity mask M |
|---|---|---|
| Normal | original aligned depth D | original M |
| Depth removed | zeroed / invalid depth | all-invalid M |
| Depth mismatched | donor depth reassigned within the recording session | donor mask moves with depth |

All three conditions use the **same trained VDF-DETR weights and the same images**; depth and its validity mask are altered at inference time. No retraining is performed, so differences are attributable to the depth input itself.

For the mismatched condition, ten one-to-one pairings are generated within the recording session using seeds 43–52. Donor depth maps and masks move together; cyclic shifting prevents a frame from retaining its own depth. Results are reported as the mean and sample standard deviation.

## Data

525 validation images / 2 550 `surfacing` boxes. The 525-image test set is used for the comparisons in Tables 3 and 4.

## Metrics

P, R, F1, AP50 and AP50–95 computed with the same unified `score()` and COCO evaluation path used for the main comparison (`tools/formal50_unified_eval.py`). Confidence was fixed at 0.50 and matching IoU was 0.50.

## Reported outcome

The normal-depth condition raises recall by **3.80 percentage points** over the depth-removed condition, while the mismatched condition falls back. This supports the interpretation that VGDF uses valid depth to recover missed detections rather than merely responding to an additional input channel.
