# Formal 50-epoch protocol

Exact settings used for the 15-model comparison (Tables 1–2 of the manuscript). Values below are recorded by `src/formal_train.py` into each run's `cfgout` dictionary and can be cross-checked against the archived `results.csv`.

## Training

| Item | Value |
|---|---|
| Epochs | 50 (fixed, no early stopping) |
| Input size | 960 × 960 |
| Batch size | 16 |
| Optimizer | AdamW (betas 0.9 / 0.999), backbone lr_scale 0.5 |
| Initial lr | 2×10⁻⁴ |
| Warmup | 2 epochs |
| LR schedule | epoch-wise **linear** decay to 5% of lr₀ (not cosine) |
| Weight decay | 1×10⁻⁴ |
| Gradient clipping | 10.0 |
| EMA | decay 0.9999, 2000 warmup steps |
| Seed | 42 (torch / random / numpy) |
| AMP | off |
| Augmentation | no mosaic / mixup / cutmix / HSV / geometric augmentation; only horizontal flip (p = 0.5) |
| Weight selection | best.pth written **only** when validation `metrics/mAP50-95(B)` strictly improves; evaluation loads the EMA weights |
| Initialization | official COCO `deimv2_pico_coco.safetensors`; **only the classification head reinitialized**; COCO tensors preserved; DEB/VGDF/RSA independently initialized |

Checkpoint selection is per-model by validation AP50–95, not by final epoch and not by test-set performance. Best epochs range from 7 to 34 across the 15 models.

## Unified evaluation

| Item | Value |
|---|---|
| COCO `maxDets` | `[1, 10, 300]` (AP uses 300 detections) |
| YOLO NMS IoU | 0.50 |
| YOLO `max_det` | 300 |
| YOLO confidence floor | 0.001 |
| DEIMv2-Pico series | end-to-end, 200-query output cap per image (not YOLO NMS) |
| P/R/F1 confidence | **per-model F1-optimal threshold**, range 0.25–0.90 (not a single fixed 0.50) |
| P/R/F1 match IoU | 0.50 |

Note that NMS IoU and P/R/F1 match IoU share the numeric value 0.50 but serve different functions. The Pico series is end-to-end; its 200-query cap is not an NMS setting.

The unified evaluation recomputes every model with the same `score()` + COCO evaluation path (`tools/formal50_unified_eval.py`). Three sets of numbers in the manuscript are intentionally distinct and must not be compared directly: per-epoch validation peaks, the `unified_best_validation.csv` recomputation, and the review-test values in Table 1.

## Reproducing

```bash
python src/formal_train.py --model pico_ab_cspatial_control --epochs 50 --batch 16 --seed 42
python tools/formal50_unified_eval.py
```

Model keys and their manuscript names are listed in the README mapping table. Set `GUIYU_ROOT`, `FORMAL_DATA` and `FORMAL_EVAL_OUTPUT` and adjust the machine-specific `ROOT` / `WORK` constants in the scripts before running; see `src/OVERVIEW.md`.
