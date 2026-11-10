# Dataset

Self-collected paired RGB-D images of juvenile mandarin fish (*Siniperca chuatsi*) in a recirculating aquaculture system (RAS), captured from a fixed top-view RGB-D camera. Single-class whole-fish bounding boxes of **surfacing** individuals (category `surfacing`). All boxes were manually reviewed before release.

## Composition

| Split | Paired RGB-D images | `surfacing` boxes | Notes |
|---|---|---|---|
| Training | 4 200 | — | not public |
| Validation | 525 | 2 550 (4.857 per image) | not public |
| Test | 525 | 2 525 | public subset |
| **Total** | **5 250** | — | single class |

The training, validation and test splits are separated by acquisition day: frames from a given day belong to exactly one split, and the test split consists of days not used for training or validation.

Each sample consists of an RGB image, the aligned depth map, and a binary **validity mask M** marking reliable depth readings. Depth values that are missing or invalid (water-surface reflection, absorption, out-of-range) are flagged by M, which is the input to the VGDF module.

Dissolved oxygen was logged **once every 10 minutes**. These are stage-level environmental records, not per-frame labels, and are not used as detection targets.

> **Important:** the depth-intervention experiment (Table 3 of the manuscript) uses a **different** 525-image validation set with 2 550 boxes. It is not the test split released here. Keep the two partitions separate when re-running the evaluation.

## Public subset

- Test split: 525 paired RGB-D images, 2 525 `surfacing` annotations, validity masks
- Format notes inside `annotations.zip`: category definition, bbox convention, depth units and the validity-mask rule
- Training (4 200) and validation (525) splits: available from the corresponding author on reasonable request
- Test subset (13 independently extractable ZIP parts + `MANIFEST.json` with SHA-256 sums): **https://github.com/2636456295a-collab/VDF-DETR/releases/tag/v1.0-testset**
- Permanent archive DOI: **to be added**

The files are attached to the repository's `v1.0-testset` GitHub Release rather than committed to git, because the subset is ~855 MB. Each ZIP extracts independently; `MANIFEST.json` lists per-part SHA-256 checksums. The git tree contains code, configurations and documentation only.

## Annotation format

- COCO JSON, single category `surfacing` (shipped in `annotations.zip` together with `test_images.txt` and format notes)
- Boxes are axis-aligned whole-fish rectangles of surfacing individuals (`xywh` in the COCO JSON)
- Surfacing individuals are identified from visible-surface cues in the RGB imagery; depth maps serve as the second input modality for detection and valid-depth fusion
- RGB and depth are paired by acquisition timestamp; the validity mask is derived from the depth channel
- Overlapping individuals are annotated separately; annotation rules are summarized in the manuscript (Section 2.2)

## Privacy and anonymization

The released files contain only experiment-required fields. Author names, affiliations, funding identifiers, camera serial numbers and tank identifiers are removed. Please check EXIF and file paths before redistributing.

## License

The released dataset subset is licensed under **CC BY 4.0** (or CC BY-NC 4.0 if commercial use is to be excluded — final choice to be confirmed). Anyone reusing it must give appropriate credit, link the license and indicate changes.

**The code in this repository is MIT-licensed; the MIT license does not apply to the data files.** The two licenses are deliberately separate.

## Suggested data citation

```
Anonymous (under review). VDF-DETR: paired RGB-D surfacing-fish detection dataset
(test subset) [Dataset]. Zenodo. DOI: [to be added].
```
