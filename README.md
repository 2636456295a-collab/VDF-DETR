# VDF-DETR

VDF-DETR is a lightweight RGB-D detector for surfacing mandarin fish (*Siniperca chuatsi*) in recirculating aquaculture. Built on DEIMv2-Pico, it combines three complementary components:

- **DEB — detail enhancement branch:** integrates shallow visual features to preserve fish contours and fine details.
- **VGDF — validity-gated depth fusion:** uses a local depth-validity mask to weight depth features during RGB-D fusion.
- **RSA — residual spatial attention:** recalibrates the fused representation at spatial locations relevant to detection.

## Detection performance

The test set contains 525 paired RGB-D images of a replacement fish group recorded on a separate day in the same system. Results use the paper's 2,521 reference boxes. Precision, recall and F1 use confidence 0.50 and matching IoU 0.50; AP follows COCO-style evaluation with up to 300 detections per image.

| Model | P (%) | R (%) | F1 (%) | AP50 (%) | AP50–95 (%) | Parameters (M) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEIMv2-Pico | 90.17 | 84.77 | 87.38 | 94.32 | 80.50 | 1.49 | 11.49 | 75.55 |
| **VDF-DETR** | **95.00** | **85.92** | **90.23** | **95.64** | **82.67** | 1.54 | 12.54 | 68.05 |

VDF-DETR improves F1 by 2.85 percentage points and AP50–95 by 2.17 points over the baseline, with 0.05 million additional parameters. FPS measures model forward passes on an NVIDIA GeForce RTX 4090 D with FP32, a batch size of one, and 960 × 960 input already on the GPU.

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

With DEB and RSA retained, VGDF improves AP50–95 by 0.38 points compared with direct depth fusion. In fixed-weight validation interventions, aligned depth improves F1 from 89.89% to 91.21% compared with zeroed depth and mask; see [`configs/depth_intervention.md`](configs/depth_intervention.md).

## Dataset and evaluation

The dataset contains 5,250 paired RGB-D images: 4,200 for training, 525 for validation and 525 for testing. The validation and test references contain 2,550 and 2,521 boxes, respectively. Training and validation data are available from the authors on reasonable request.

The RGB-D test images and depth maps are available in the [`v1.0-testset` release](https://github.com/2636456295a-collab/VDF-DETR/releases/tag/v1.0-testset). The release's accompanying COCO annotation file contains 2,525 boxes; use [`data/test_reference_2521.json`](data/test_reference_2521.json) for the 2,521-box evaluation reported above. The JSON maps each image ID to a list of `[x1, y1, x2, y2]` reference boxes in pixel coordinates.

To evaluate saved prediction records against the paper's references:

```bash
python tools/evaluate_test_predictions.py --predictions predictions.json --references data/test_reference_2521.json --confidence 0.50
```

The command requires `pycocotools` and saved prediction records in the format described in [`configs/formal50.md`](configs/formal50.md). It evaluates saved predictions; the command does not itself generate predictions or train a model.

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

Code is provided under the [MIT license](LICENSE). The released dataset subset is CC BY 4.0. Please cite the repository when using the code or data; article citation details will be added when available. See [`CITATION.cff`](CITATION.cff).
