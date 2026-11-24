# Depth-input intervention protocol

Corresponding to Sections 2.5 and 3.3 and Table 5 of the manuscript.

## Purpose

Assess the contribution of depth-input availability and RGB–depth correspondence to surfacing-fish detection with RGB inputs and model weights held fixed.

## Conditions

| Condition | Depth input | Validity mask M |
|---|---|---|
| Normal | original aligned depth D | original M |
| Depth removed | zeroed / invalid depth | all-invalid M |
| Depth mismatched | donor depth reassigned within the recording session | donor mask moves with depth |

All three conditions use the **same trained VDF-DETR weights and the same images**; depth and its validity mask are altered at inference time. The comparison measures the response of the fixed-weight detector to the three depth-and-mask input conditions.

For the mismatched condition, ten one-to-one pairings are generated within the recording session using seeds 43–52. Donor depth maps and masks move together; cyclic shifting prevents a frame from retaining its own depth. Results are reported as the mean and sample standard deviation.

## Metrics

P, R, F1, AP50 and AP50–95 computed with the same unified `score()` and COCO evaluation path used for the main comparison (`tools/formal50_unified_eval.py`). Confidence was fixed at 0.50 and matching IoU was 0.50.

## Reported outcome

Compared with zeroed depth and mask, aligned depth improves recall, F1, AP50 and AP50–95 by 3.80, 1.32, 0.69 and 1.26 percentage points, respectively. Compared with the mean of ten mismatched pairings, aligned depth improves F1 and AP50–95 by 0.39 and 0.34 points, respectively. These results characterize the contribution of depth availability and RGB–depth correspondence in the trained detector.
