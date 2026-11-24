# Formal 50-epoch protocol

Pico-series training and evaluation settings for the 12-model comparison and ablation study (Tables 3 and 4 of the manuscript).

## Training

| Item | Value |
|---|---|
| Epochs | 50 (fixed, no early stopping) |
| Input size | 960 × 960 |
| Batch size | 16 |
| Gradient accumulation | 1 |
| Optimizer | AdamW (betas 0.9 / 0.999), backbone lr_scale 0.5 |
| Initial lr | 2×10⁻⁴ |
| Warmup | 2 epochs |
| LR schedule | epoch-wise linear decay to 5% of lr₀ |
| Weight decay | 1×10⁻⁴ |
| Gradient clipping | 10.0 |
| EMA | decay 0.9999, 2000 warmup steps |
| Seed | 42 (torch / random / numpy) |
| AMP | off |
| Augmentation | horizontal flip (p = 0.5); mosaic, mixup, cutmix, HSV and other geometric augmentations disabled |
| Weight selection | best.pth written **only** when validation `metrics/mAP50-95(B)` strictly improves; evaluation loads the EMA weights |
| Initialization | official COCO `deimv2_pico_coco.safetensors`; **only the classification head reinitialized**; COCO tensors preserved; DEB/VGDF/RSA independently initialized |

Pico-series checkpoint selection is based on validation AP50–95.

## Unified evaluation

| Item | Value |
|---|---|
| COCO `maxDets` | `[1, 10, 300]` (AP uses 300 detections) |
| YOLO NMS IoU | 0.50 |
| YOLO `max_det` | 300 |
| YOLO confidence floor | 0.001 |
| DEIMv2-Pico series | end-to-end, 200-query output cap per image (not YOLO NMS) |
| P/R/F1 confidence | 0.50 for the test results in Tables 3 and 4 |
| P/R/F1 match IoU | 0.50 |

Note that NMS IoU and P/R/F1 match IoU share the numeric value 0.50 but serve different functions. The Pico series is end-to-end; its 200-query cap is not an NMS setting.

The manuscript reports test results using the same matching and COCO evaluation definitions across models. The repository also contains validation evaluation utilities.

## Reproducing

```bash
python src/formal_train.py --model pico_ab_cspatial_control --epochs 50 --batch 16 --accum 1 --seed 42
# Recompute validation metrics. For test metrics from saved predictions, see the command below.
python tools/formal50_unified_eval.py
```

Model keys and their manuscript names are listed in the README mapping table. Set `GUIYU_ROOT`, `FORMAL_DATA` and `FORMAL_EVAL_OUTPUT` and adjust the machine-specific `ROOT` / `WORK` constants in the scripts before running; see `src/OVERVIEW.md`.

## Test metrics from saved predictions

`tools/evaluate_test_predictions.py` computes P/R/F1 and COCO AP on the 525-image, 2,521-box paper test references from saved prediction records. It requires `pycocotools`, an image-ID to xyxy reference-box JSON mapping, and a JSON list of per-image prediction records with `id`, `width`, `height`, and `predictions` entries containing `xyxy` and `confidence`. The paper tables use confidence 0.50. This tool evaluates saved predictions; it does not run inference or convert the released COCO annotations.

```bash
python tools/evaluate_test_predictions.py --predictions predictions.json --references data/test_reference_2521.json --confidence 0.50
```
