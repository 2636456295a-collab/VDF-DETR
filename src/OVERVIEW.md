# Source file overview

The repository provides research implementations of VDF-DETR, including the model definitions, training scripts and evaluation utilities for the formal 50-epoch experiments.

| File | Role |
|---|---|
| `wall20_models_v1.py` | A/B module definitions and model builder. `Encoder.fuse()` implements DEB (`use_detail`) and VGDF (`depth_mode='mask'`), including the `[D⊙M, M]` concatenation and valid-depth-ratio gate. |
| `models_spatial.py` | RSA (`ResidualSpatialAttention`, 99 parameters, γ = 0) and the VDF-DETR builder. Contains self-checks for initialization and parameter count. |
| `short20_deim_core.py` | DEIMv2-Pico baseline adapter: official model/criterion loading, decoding (confidence floor 0.001), 200-query output cap, data/target conversion. |
| `short20_common.py` | Shared IO, retry and job utilities. |
| `formal_train.py` | Standalone trainer reproducing the exact loop (AdamW, backbone lr_scale 0.5, EMA 0.9999 / 2000 warmups, epoch-wise linear decay, clip 10, fliplr 0.5) without the legacy queue dependencies. Writes the full hyperparameter record into each run's config. |
| `../tools/formal50_unified_eval.py` | Recomputes validation metrics from available predictions. |
| `../tools/cloud_unified_validation.py` | Corresponding validation metric recomputation. |
| `../tools/evaluate_test_predictions.py` | Computes P/R/F1 and COCO AP from saved predictions and reference boxes. |

## Internal keys

`SPECS` in `wall20_models_v1.py` defines `pico`, `pico_a`, `pico_b`, `pico_ab`, `pico_a_depth`, `pico_a_mask`. The paper configurations are registered in `models_spatial.py` (`pico_ab_cspatial_control`) and in the training/evaluation scripts (`pico_b_mask`, `pico_a`, `pico_a_mask`).

## Path dependencies

The scripts were written for two machines and contain hard-coded roots:

- local: `F:/GUIYU` and `ROOT = Path(__file__).resolve().parents[1]`
- cloud: `/home/waas/GUIYU`, `GUIYU_ROOT`, `FORMAL_DATA`, `FORMAL_EVAL_OUTPUT`

Before running, set those environment variables or edit the `ROOT` / `WORK` / `JOB` constants. Obtain `deimv2_pico_coco.safetensors` from the upstream DEIMv2 release.

## Dependencies

The research scripts list Python 3.9, PyTorch, upstream DEIMv2 / D-FINE and RT-DETR sources, Ultralytics (for YOLO baselines), and `faster_coco_eval` as dependencies. The manuscript reports Python 3.10.14, PyTorch 2.6.0 and CUDA 12.6 for the experimental environment. Use the archived run manifests to match the actual environment of each experiment. The saved-prediction test evaluator requires `pycocotools`.
