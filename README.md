# VDF-DETR: RGB-D detection and spatial analysis of surfacing mandarin fish under declining dissolved oxygen

VDF-DETR is a lightweight RGB-D detector for surfacing mandarin fish (*Siniperca chuatsi*) in recirculating aquaculture. Built on DEIMv2-Pico, it combines three complementary components:

- **DEB — detail enhancement branch:** integrates shallow visual features to preserve fish contours and fine details.
- **VGDF — validity-gated depth fusion:** weights depth features according to the local proportion of valid-range depth readings during RGB-D fusion.
- **RSA — residual spatial attention:** recalibrates the fused representation at spatial locations relevant to detection.

## Detection performance

Precision, recall and F1 use confidence 0.50 and matching IoU 0.50; AP follows COCO-style evaluation with up to 300 detections per image.

| Model | P (%) | R (%) | F1 (%) | AP50 (%) | AP50–95 (%) | Parameters (M) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEIMv2-Pico | 90.17 | 84.77 | 87.38 | 94.32 | 80.50 | 1.49 | 11.49 | 75.55 |
| YOLOv8n | 90.09 | 83.66 | 86.75 | 92.18 | 73.84 | 3.01 | 18.43 | 166.11 |
| YOLOv10n | 82.38 | 76.79 | 79.49 | 88.35 | 71.07 | 2.71 | 19.19 | 100.89 |
| YOLO11n | 82.92 | 82.03 | 82.47 | 90.45 | 72.95 | 2.59 | 14.80 | 116.30 |
| YOLO12n | 70.79 | 84.29 | 76.95 | 88.02 | 71.06 | 2.57 | 19.55 | 73.40 |
| D-FINE-N | 78.54 | 85.36 | 81.81 | 90.81 | 77.77 | 3.73 | 15.92 | 65.55 |
| RT-DETRv2-S | 75.35 | 91.79 | 82.76 | 92.18 | 75.61 | 20.08 | 135.95 | 90.70 |
| Faster R-CNN | 68.36 | 86.31 | 76.30 | 86.09 | 69.72 | 41.35 | 375.08 | 68.75 |
| RetinaNet | 72.71 | 75.45 | 74.05 | 79.58 | 63.54 | 32.22 | 366.88 | 72.10 |
| RT-DETR-R18 | 79.04 | 93.18 | 85.53 | 93.78 | 76.94 | 20.08 | 135.95 | 69.72 |
| Deformable DETR | 92.22 | 74.30 | 82.29 | 92.14 | 73.13 | 40.05 | 351.87 | 33.57 |
| **VDF-DETR** | **95.00** | 85.92 | **90.23** | **95.64** | **82.67** | 1.54 | 12.54 | 68.05 |

VDF-DETR improves F1 by 2.85 percentage points and AP50–95 by 2.17 points over the baseline, with 0.05 million additional parameters. FPS measures 40 model forward passes after 10 warm-up passes on an NVIDIA GeForce RTX 4090 D with FP32, a batch size of one, and 960 × 960 input already on the GPU. VDF-DETR uses RGB-D inputs; the other listed detectors use RGB inputs.

## Ablation study

| Configuration | P (%) | R (%) | F1 (%) | AP50 (%) | AP50–95 (%) |
|---|---:|---:|---:|---:|---:|
| DEIMv2-Pico | 90.17 | 84.77 | 87.38 | 94.32 | 80.50 |
| Pico + direct depth | 90.24 | 84.73 | 87.40 | 94.42 | 80.59 |
| Pico + VGDF | 89.91 | 85.16 | 87.47 | 94.01 | 80.43 |
| Pico + DEB | 91.35 | 87.50 | 89.38 | 95.07 | 82.37 |
| Pico + DEB + VGDF | 94.19 | 86.12 | 89.97 | 95.55 | 82.60 |
| Pico + DEB + direct depth + RSA | 92.47 | 87.62 | 89.98 | 95.44 | 82.29 |
| **VDF-DETR** | **95.00** | 85.92 | **90.23** | **95.64** | **82.67** |

With DEB and RSA retained, VGDF improves precision by 2.53 percentage points and AP50–95 by 0.38 points compared with direct depth fusion.

## Depth-input comparison

The same VDF-DETR checkpoint was evaluated with three depth-input conditions on the validation split:

| Condition | P (%) | R (%) | F1 (%) | AP50 (%) | AP50–95 (%) |
|---|---:|---:|---:|---:|---:|
| Aligned depth | 92.71 | 89.76 | 91.21 | 96.18 | 84.69 |
| Depth and mask zeroed | 94.20 | 85.96 | 89.89 | 95.49 | 83.43 |
| Mismatched depth | 92.99 ± 0.14 | 88.75 ± 0.16 | 90.82 ± 0.10 | 95.93 ± 0.03 | 84.35 ± 0.05 |

Mismatched values are the mean ± sample standard deviation over ten pairings. See [`configs/depth_intervention.md`](configs/depth_intervention.md) for the protocol.

## Surfacing activity and dissolved oxygen

Spatial analysis found that 97.86% of surfacing-fish box centres were in a near-wall band covering approximately 27.75% of the water surface. Mean annotated fish counts per frame were negatively associated with dissolved oxygen (Spearman's ρ = −0.848).

## Implementation

| Component | Configuration / implementation |
|---|---|
| VDF-DETR | `pico_ab_cspatial_control`; `src/models_spatial.py`, `src/wall20_models_v1.py` |
| DEIMv2-Pico | `pico`; `src/short20_deim_core.py` |
| DEB | `use_detail=True`; `src/wall20_models_v1.py` |
| VGDF | `depth_mode='mask'`; `src/wall20_models_v1.py` |
| RSA | `ResidualSpatialAttention`; `src/models_spatial.py` |

The Pico-series configurations use COCO-pretrained DEIMv2-Pico initialization, 960 × 960 inputs, and checkpoint selection based on validation AP50–95. For training configuration and evaluation details see [`configs/formal50.md`](configs/formal50.md) and [`src/OVERVIEW.md`](src/OVERVIEW.md).

## License and citation

Code is provided under the [MIT license](LICENSE). Please cite the repository when using the code; article citation details will be added when available. See [`CITATION.cff`](CITATION.cff).
