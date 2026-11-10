# Dataset

Self-collected paired RGB-D images of juvenile mandarin fish (*Siniperca chuatsi*) in a recirculating aquaculture system (RAS), captured from a fixed top-view RGB-D camera. Single-class **whole-fish bounding boxes**. This is a detection dataset; it contains **no behavioral or surface-breathing labels**.

## Composition

| Split | Paired RGB-D images | Whole-fish boxes | Notes |
|---|---|---|---|
| Training | 4 200 | — | not public |
| Validation | 525 | 2 550 (4.857 per image) | not public |
| Review test | 525 | 2 525 | public subset |
| **Total** | **5 250** | — | single class |

Each sample consists of an RGB image, the aligned depth map, and a binary **validity mask M** marking reliable depth readings. Depth values that are missing or invalid (water-surface reflection, absorption, out-of-range) are flagged by M, which is the input to the VGDF module.

Dissolved oxygen was logged **once every 10 minutes**. These are stage-level environmental records, not per-frame labels, and are not used as detection targets.

> **Important:** the depth-intervention experiment (Table 3 of the manuscript) uses a **different** 525-image validation set with 2 550 boxes. It is not the review-test split released here. Keep the two partitions separate when re-running the evaluation.

## Public subset

- Review test split: 525 paired RGB-D images, 2 525 whole-fish annotations, validity masks
- 50 demonstration samples drawn from the training/validation splits, illustrating annotation format, RGB-D pairing and the appearance of missing depth
- Training (4 200) and validation (525) splits: available from the corresponding author on reasonable request
- Permanent archive and DOI: **to be added**

The files are hosted outside this repository because a full paired RGB-D subset exceeds GitHub's practical file-size limits. This repository contains code, configurations and data documentation only.

## Annotation format

- COCO JSON, single category `mandarin_fish`
- Boxes are axis-aligned whole-fish rectangles (`xyxy` / `xywh` as specified in the JSON)
- RGB and depth are paired by acquisition timestamp; the validity mask is derived from the depth channel
- Overlapping individuals are annotated separately; annotation rules are summarized in the manuscript (Section 2.2)

## Privacy and anonymization

The released files contain only experiment-required fields. Author names, affiliations, funding identifiers, camera serial numbers and tank identifiers are removed. Please check EXIF and file paths before redistributing.

## License

The released dataset subset is licensed under **CC BY 4.0** (or CC BY-NC 4.0 if commercial use is to be excluded — final choice to be confirmed). Anyone reusing it must give appropriate credit, link the license and indicate changes.

**The code in this repository is MIT-licensed; the MIT license does not apply to the data files.** The two licenses are deliberately separate.

## Suggested data citation

```
Anonymous (under review). VDF-DETR: paired RGB-D mandarin fish detection dataset
(review-test subset and demonstration samples) [Dataset]. Zenodo. DOI: [to be added].
```
