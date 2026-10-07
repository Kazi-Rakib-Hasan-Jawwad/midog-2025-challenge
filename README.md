# MIDOG 2025 — DeepLabV3+FDA Aug

Recovered code and Docker submission sources for **C-Omics**, submitted by **krhasan02** to the **MIDOG 2025 Track 1: Mitosis Detection** challenge.

**Final leaderboard: 14th of 15 published entries · F1 0.6336 · precision 0.7228 · recall 0.5639.** Verified against the [official final leaderboard](https://midog2025.grand-challenge.org/evaluation/track-1-mitosis-detection-final-submission-phase/leaderboard/) on 7 October 2026. The [final evaluation](https://midog2025.grand-challenge.org/evaluation/6f3f6c11-3ae0-47ec-99f6-eafed0bf45b1/) identifies the algorithm as **DeepLabV3+FDA Aug**.

The approach uses a modified DeepLabv3+ network with a ResNet50 backbone, segmentation of background/mitotic/non-mitotic pixels, training augmentation including Fourier Domain Adaptation, and sliding-window inference with point extraction. Authors of the accompanying contribution: **Kazi Rakib Hasan, Sijin Kim, and Junghwan Cho**.

## Repository map

| Location | Contents |
| --- | --- |
| [`docker/midog`](docker/midog) | Recovered inference build context with GroupNorm conversion |
| [`docker/midog_v2`](docker/midog_v2) | Alternate BatchNorm build context and original `training_resources` bundle |
| [`training/experiment_workspace`](training/experiment_workspace) | Training code, network variants, saved run-associated source and hyperparameters |
| [`training/preprocessing_workspace`](training/preprocessing_workspace) | Patch/mask preparation, stain augmentation, earlier training code and reference implementations |
| [`evaluation`](evaluation) | Recovered evaluation workspace and organizer reference copies |
| [`archive/early_challenge`](archive/early_challenge) | Early challenge implementation |
| [`archive/desktop_experiments`](archive/desktop_experiments) | Additional training/sampling fragments and four scalar training-curve exports |
| [`archive/docker_images`](archive/docker_images) | Source recovered from five existing Docker images, identified by image digest |
| [`docs`](docs) | Final and preliminary results, file provenance, artifact index and reproduction notes |

This is a preservation and publication project assembled from the surviving workspaces. Original implementation files are retained byte for byte and identified in [`docs/source_manifest.json`](docs/source_manifest.json). Several historical versions are included; the local image tags and directory names alone do **not** establish which exact bytes Grand Challenge executed. See [reproduction notes](docs/REPRODUCIBILITY.md).

## Run the preserved inference container

The build contexts use the historical CUDA 11.1 / PyTorch 1.9 / Python 3.8 stack. Model weights are separate from the container and are not committed. Obtain the matching `last.ckpt` from the project owner; [artifact metadata](docs/ARTIFACTS.md) identifies the recovered candidates.

From this repository:

```bash
docker build -t midog-2025:gn docker/midog

# Set these to existing absolute paths. MODEL_DIR must contain last.ckpt.
export MODEL_DIR=/absolute/path/to/model
export INPUT_DIR=/absolute/path/to/input
export OUTPUT_DIR=/absolute/path/to/output
mkdir -p "$OUTPUT_DIR"

docker run --rm --gpus all --network none \
  --mount "type=bind,src=$MODEL_DIR,dst=/opt/ml/model,readonly" \
  --mount "type=bind,src=$INPUT_DIR,dst=/input,readonly" \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/output" \
  midog-2025:gn
```

The container runs as user `algo`; ensure the output directory is writable by that container user. The input may use the Grand Challenge socket layout or contain one supported image directly. Output is `mitotic-figures.json`. For the BatchNorm candidate, build `docker/midog_v2` with a separate tag and use its matching checkpoint. Building and complete model inference were **not** rerun during recovery.

## Training and validation

The recovered training scripts retain their original data paths, GPU selection and experiment settings. They are historical source snapshots, not a newly standardized training CLI. Read [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before using them. Obtain datasets through their official distribution; no images, ground-truth annotations, model weights or private evaluation exports are included here.

Run the dependency-free preservation checks with Python 3.10 or newer:

```bash
python scripts/validate_repository.py
```

The local recovery used the owner's existing Python 3.10.11 working environment without installing or changing packages. The checks validate source checksums, Python syntax, JSON structure and publication exclusions; they do not establish model accuracy.

## Results and citation

The final F1 is **0.6336**. The contribution paper's **0.654** refers to the **preliminary** evaluation. Full displayed final metrics and both recovered preliminary aggregate results are in [RESULTS.md](docs/RESULTS.md).

Contribution: *A DeepLabv3+ Approach for Mitotic Figure Detection in the MIDOG 2025 Challenge* — Kazi Rakib Hasan, Sijin Kim, Junghwan Cho (2025). [DOI: 10.5281/zenodo.17017920](https://doi.org/10.5281/zenodo.17017920).

The repository uses the [Apache 2.0 license](LICENSE) selected by the project owner. Existing third-party attributions are retained; see [third-party notes](docs/THIRD_PARTY.md) for component sources and terms.
