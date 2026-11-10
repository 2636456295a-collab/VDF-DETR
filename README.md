# VDF-DETR
> **Status:** code accompanying a manuscript currently under double-blind peer review. Author names and affiliations are intentionally omitted and will be added on acceptance.

Official implementation of **VDF-DETR**, a lightweight RGB-D detector for single-class surfacing-fish detection in recirculating aquaculture systems (RAS). VDF-DETR extends **DEIMv2-Pico** with three modules: **DEB** (detail enhancement branch), **VGDF** (valid-depth fusion) and **RSA** (residual spatial attention), reaching **1.54 M parameters / 12.54 GFLOPs**.

## Main results

Unified test set (525 paired RGB-D images, 2 525 `surfacing` boxes):

| Model | P (%) | R (%) | F1 (%) | AP50 (%) | AP50–95 (%) |
|---|---|---|---|---|---|
| VDF-DETR | 93.24 | 87.96 | 90.52 | 95.68 | 82.71 |

Gains over the DEIMv2-Pico baseline: +2.90 / +3.17 / +3.04 / +1.32 / +2.21 percentage points respectively. The 15-model comparison and the 5-configuration ablation are reported in Tables 1–2 of the manuscript.

## Manuscript ↔ code mapping

The experiment scripts use internal configuration keys. This table maps them to the names used in the paper.

| Manuscript | Experiment key | Implementation |
|---|---|---|
| **VDF-DETR** | `pico_ab_cspatial_control` | `src/models_spatial.py` + `src/wall20_models_v1.py` |
| DEIMv2-Pico (baseline) | `pico` | `src/short20_deim_core.py::make_model` |
| Pico + DEB | `pico_a` | `src/wall20_models_v1.py` |
| Pico + VGDF | `pico_b_mask` | `src/wall20_models_v1.py` |
| Pico + DEB + VGDF | `pico_a_mask` | `src/wall20_models_v1.py` |
| DEB | `use_detail=True` | shallow detail residual, `wall20_models_v1.py` L29–31, L40 |
| VGDF | `depth_mode='mask'` | `[D⊙M, M]` encoded by `depth_branch`, `wall20_models_v1.py` L41–52 |
| RSA | `ResidualSpatialAttention` | 99 parameters, γ initialized to 0, `models_spatial.py` L12–20 |

VGDF concatenates the masked depth `D⊙M` with the binary validity mask `M` before the lightweight depth encoder, and uses the valid-depth ratio as the fusion gate. RSA applies `F·(1 + tanh(γ)·(2σ(conv₇([mean_c(F), max_c(F)])) − 1))` after the A+B fusion. The final VDF-DETR has **no `quality_gate`** (asserted in `models_spatial.py`); the `quality` branches (`pico_b`, `pico_ab`) are earlier exploratory variants and are **not** used in the reported results.

## Repository layout

```
src/    model definition, A/B/C modules, data interface, standalone trainer
tools/  unified evaluation on validation and test splits
configs/formal50.md            exact 50-epoch training and evaluation protocol
configs/depth_intervention.md  depth-input intervention conditions (Section 3.4)
DATASET.md                     data composition, public subset, annotation format, license
CITATION.cff          citation metadata
```

See `src/OVERVIEW.md` for a per-file description and the machine-specific path constants that need adjusting before running.

## Training and evaluation

All protocol values (50 epochs, 960² input, batch 16, AdamW, lr₀ = 2×10⁻⁴, 2-epoch warmup, epoch-wise linear decay to 5%, weight decay 10⁻⁴, seed 42, EMA 0.9999, selection by validation AP50–95) are recorded in `configs/formal50.md`. Evaluation uses COCO `maxDets = [1, 10, 300]`; for YOLO models NMS IoU = 0.50, `max_det = 300`, confidence floor 0.001; DEIMv2-Pico series are end-to-end with a 200-query output cap; P/R/F1 use per-model F1-optimal confidence thresholds (0.25–0.90) at match IoU 0.50.

## Initialization

Fine-tuned from the official COCO weight `deimv2_pico_coco.safetensors` (obtain from upstream; not redistributed here). Only the classification head is reinitialized; all COCO backbone tensors are preserved bit-for-bit. DEB/VGDF/RSA are independently initialized, so all 15 compared models have explicitly documented initialization sources.

## Data

The test subset (525 paired RGB-D images, 2 525 `surfacing` annotations, depth maps and validity masks) is attached to the [`v1.0-testset` release](https://github.com/2636456295a-collab/VDF-DETR/releases/tag/v1.0-testset); training and validation splits are available on reasonable request. The three splits are separated by acquisition day. See `DATASET.md`.

## License

Code: **MIT** (see `LICENSE`). Dataset subset: **CC BY 4.0** (see `DATASET.md`). The two licenses are separate; the MIT license does not cover the released data files.

## Citation

The article is under review. Until publication, please cite this repository and the dataset DOI (to be added). A `CITATION.cff` is provided.
