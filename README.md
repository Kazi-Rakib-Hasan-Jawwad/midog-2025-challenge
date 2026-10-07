# MIDOG 2025 — DeepLabV3+FDA Aug

Code for **C-Omics**, submitted by **krhasan02** to **MIDOG 2025 Track 1: Mitosis Detection**. The approach uses DeepLabv3+ with a ResNet50 backbone, Fourier Domain Adaptation during training, and sliding-window inference with mitotic-point extraction.

**Final leaderboard: 14th of 15 published entries · F1 0.6336 · precision 0.7228 · recall 0.5639.** [Official leaderboard](https://midog2025.grand-challenge.org/evaluation/track-1-mitosis-detection-final-submission-phase/leaderboard/) · [Final evaluation](https://midog2025.grand-challenge.org/evaluation/6f3f6c11-3ae0-47ec-99f6-eafed0bf45b1/). Checked on 7 October 2026.

## Code

| Directory | Purpose |
| --- | --- |
| [preprocessing/](preprocessing/README.md) | Build 512 × 512 image/mask patches from raw MIDOG++ |
| [training/](training/) | Main training script, loader, augmentation, loss and network |
| [docker/midog/](docker/midog/) | Inference container with GroupNorm conversion |
| [docker/midog_v2/](docker/midog_v2/) | Alternate inference container retaining BatchNorm |

## Prepare the dataset

Use Python 3.10 or newer with the dependencies in [preprocessing/requirements.txt](preprocessing/requirements.txt). Obtain MIDOG++ through its [official distribution](https://github.com/DeepMicroscopy/MIDOGpp).

```bash
python preprocessing/rebuild_patches.py build \
  --source /path/to/MIDOGpp-main \
  --split-csv /path/to/datasets_xvalidation.csv \
  --output /path/to/rebuilt/512_seg_root
```

The output contains paired `img/` and `mask/` directories. Labels are **0 background, 1 mitosis, 2 non-mitosis**. See the [preprocessing guide](preprocessing/README.md) for the recipe and output manifests.

The local reconstruction produced **15,758 patch pairs from 503 raw TIFFs** and passed independent pixel validation and a loader compatibility check. This repository contains the exact builder used for that reconstruction. The original patch inventory was unavailable, so byte-identical recovery of the historical training set is not established.

## Training

The entry point is [training/Train.py](training/Train.py). The recovered main training files retain their original behavior: absolute paths, GPU index 2, a one-epoch continuation and loading of an existing checkpoint. Configure these for your environment before running. The historical [requirements](training/requirements.txt) are a reference environment, not a verified installation for current Python/PyTorch versions.

The data module currently splits patches randomly and uses the full image pool for FDA references. For a new experiment, use the generated image-level split manifests and keep validation images and FDA references separate from training. Saved normalization constants are retained. Training and model accuracy were not rerun during this dataset rebuild.

## Docker inference

The containers use the historical CUDA 11.1 / PyTorch 1.9 / Python 3.8 stack. Obtain the matching `last.ckpt` separately from the project owner. Keep each network variant with its corresponding checkpoint; the exact local container/checkpoint corresponding to the submitted Grand Challenge version has not been established.

```bash
docker build -t midog-2025:gn docker/midog

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

`MODEL_DIR` must contain `last.ckpt`, and the output directory must be writable by the container's `algo` user. Output is `mitotic-figures.json`. For BatchNorm, build `docker/midog_v2` with its matching checkpoint. Recovered inference defaults use tile size 1024, stride 614 and mitotic threshold 0.5; command-line defaults take precedence over saved YAML. Fresh Docker builds and full inference remain unverified.

## Citation and license

Kazi Rakib Hasan, Sijin Kim, Junghwan Cho. *A DeepLabv3+ Approach for Mitotic Figure Detection in the MIDOG 2025 Challenge* (2025). [DOI: 10.5281/zenodo.17017920](https://doi.org/10.5281/zenodo.17017920). The paper's F1 of 0.654 is from the preliminary evaluation; the final F1 is 0.6336.

[Apache 2.0](LICENSE). Existing [third-party attributions](THIRD_PARTY.md) are retained. Dataset images, annotations, weights, logs, inventories and verification tooling are maintained outside this repository.
