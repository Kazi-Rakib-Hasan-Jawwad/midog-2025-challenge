# MIDOG++ patch preparation

`rebuild_patches.py` is the exact builder used to reconstruct the local 512-pixel dataset. It combines the surviving grid and mask conventions with the 2025 contribution paper's filtering rules. The original combined generator and patch inventory were unavailable; this is a documented reconstruction.

## Inputs and commands

Use Python 3.10+ and the packages in [requirements.txt](requirements.txt). The source directory must contain `images/*.tiff` and `databases/MIDOG++.json`. Supply the upstream `datasets_xvalidation.csv` containing `Slide;Dataset` assignments.

```bash
python preprocessing/rebuild_patches.py plan \
  --source /path/to/MIDOGpp-main \
  --split-csv /path/to/datasets_xvalidation.csv \
  --output /path/to/rebuild_plan.json

python preprocessing/rebuild_patches.py build \
  --source /path/to/MIDOGpp-main \
  --split-csv /path/to/datasets_xvalidation.csv \
  --output /path/to/rebuilt/512_seg_root
```

Keep output outside the raw source directory. Unrelated nonempty output directories are rejected. Repeating the build resumes completed images after checking patch hashes and the source/software/recipe binding. `--image-ids` supports a small initial subset. Allow at least 50 GiB of free space for the builder's storage reserve; the completed image/mask files occupy about 7.77 GiB.

## Recipe

- Use the 503 annotated TIFFs at original resolution. The JSON's 50 unavailable, unannotated image records are explicitly excluded.
- Interpret bounding boxes as **xyxy corners**, not COCO xywh. The recovered annotations contain 50 × 50 boxes.
- Build an even coverage grid with `n = round(1 + dimension / 512)` positions per axis, spread from zero to `dimension - 512` using exact rational rounding. Every patch is 512 × 512.
- Skip empty patches. Reject a patch if any included object's visible area is below **80%** of its original box, or if valid source-image coverage is below **80%**. Transparent pixels and geometric padding do not count as source coverage; no tissue-intensity threshold is applied.
- Rasterize boxes in annotation order using OpenCV's inclusive endpoints and integer truncation. Later overlapping annotations overwrite earlier ones. Even endpoint-only raster touches enter the visibility filter.
- Save lossless RGB PNG images and single-channel uint8 masks with matching filenames: **0 background, 1 mitosis, 2 non-mitosis**. These are box-derived semantic targets, not traced cell boundaries.
- Apply no resizing, color normalization, augmentation, balancing or random sampling.

## Output

| Path | Contents |
| --- | --- |
| `img/`, `mask/` | Matched image/mask PNG pairs |
| `patch_manifest.csv` | Source image, coordinates, labels, split and patch hashes |
| `candidate_decisions.csv` | Retention/rejection decisions for all windows |
| `annotation_coverage.csv` | Coverage of every annotation, including omissions |
| `image_groups.csv`, `splits/` | Upstream image-level train/test assignments |
| `metadata/slides/` | Source hashes and resumable per-image receipts |
| `build_config.json`, `build_summary.json` | Input binding, recipe, counts and measured channel statistics |

The completed local build produced **15,758 pairs**: 5,354 mitosis-only, 8,465 non-mitosis-only and 1,939 mixed patches. Of 80,829 candidate windows, 56,671 were empty and 8,400 failed object visibility. Filtering leaves **6,339 of 26,286 annotations unrepresented**, recorded explicitly in the coverage table.

The upstream assignments contain 392 training and 111 test images, yielding 12,333 and 3,425 patches. These are MIDOG++ reference splits, not the hidden challenge test set. The flat output includes both; a new training experiment must respect image groups to prevent overlapping patches from crossing partitions.

Every saved patch in the local build passed independent image/mask pixel comparison with the raw source, and the training loader read the dataset successfully. Verification scripts, tests, logs and detailed execution receipts stay in the local collection. The builder reports `built_pending_independent_validation`; it does not certify its own output or generate a validation completion marker.
