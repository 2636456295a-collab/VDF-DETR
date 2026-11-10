# Source file overview

These are the original research scripts used for the formal 50-epoch experiments. They are provided for verification of the method described in the manuscript, not as a packaged installable library.

| File | Role |
|---|---|
| `wall20_models_v1.py` | A/B module definitions and model builder. `Encoder.fuse()` implements DEB (`use_detail`) and VGDF (`depth_mode='mask'`), including the `[D⊙M, M]` concatenation and valid-depth-ratio gate. |
| `models_spatial.py` | RSA (`ResidualSpatialAttention`, 99 parameters, γ = 0) and the VDF-DETR builder. Contains self-checks asserting the C module is an identity transform at initialization, preserves A/B tensors and RNG streams exactly, and that no `quality_gate` exists. Asserts total parameters = 1 542 854. |
| `short20_deim_core.py` | DEIMv2-Pico baseline adapter: official model/criterion loading, decoding (confidence floor 0.001), 200-query output cap, data/target conversion. |
| `short20_common.py` | Shared IO, retry and job utilities. |
| `formal_train.py` | Standalone trainer reproducing the exact loop (AdamW, backbone lr_scale 0.5, EMA 0.9999 / 2000 warmups, epoch-wise linear decay, clip 10, fliplr 0.5) without the legacy queue dependencies. Writes the full hyperparameter record into each run's config. |
| `../tools/formal50_unified_eval.py` | Unified recomputation of COCO AP and P/R/F1 for all 15 models on the review test split. |
| `../tools/cloud_unified_validation.py` | Corresponding recomputation on the 525-frame validation split. |

## Internal keys

`SPECS` in `wall20_models_v1.py` defines `pico`, `pico_a`, `pico_b`, `pico_ab`, `pico_a_depth`, `pico_a_mask`. The final configurations are registered in `models_spatial.py` (`pico_ab_cspatial_control`) and in the training/evaluation scripts (`pico_b_mask`, `pico_a`, `pico_a_mask`). **Keys containing `quality` refer to earlier exploratory gated variants and are not part of the reported results.**

## Path dependencies

The scripts were written for two machines and contain hard-coded roots:

- local: `F:/GUIYU` and `ROOT = Path(__file__).resolve().parents[1]`
- cloud: `/home/waas/GUIYU`, `GUIYU_ROOT`, `FORMAL_DATA`, `FORMAL_EVAL_OUTPUT`

Before running, set those environment variables or edit the `ROOT` / `WORK` / `JOB` constants. No dataset images, annotations or pretrained weights are shipped in this repository; obtain `deimv2_pico_coco.safetensors` from the upstream DEIMv2 release and prepare the paired RGB-D dataset as described in `DATASET.md`.

## Dependencies

Python 3.9, PyTorch, the upstream DEIMv2 / D-FINE and RT-DETR sources, Ultralytics (for YOLO baselines), `faster_coco_eval`. Exact package versions are recorded in the archived run manifests; pin your environment to those before comparing numbers.
