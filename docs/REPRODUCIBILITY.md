# Reproduction status

## Preserved evidence

The repository contains the two surviving Docker build contexts and source extracted from five local Docker images. `source_manifest.json` records source-relative provenance, SHA-256 and byte size. `docker_images.json` records the immutable image IDs, tags and creation timestamps. Source extraction used temporary **stopped** containers; no training or inference container was started.

The final Grand Challenge evaluation used:

- Evaluation: `6f3f6c11-3ae0-47ec-99f6-eafed0bf45b1`
- Submission: `f17119a5-e2c3-40d3-9b5d-2895df752958`
- Algorithm image version: `12d354b5-d6ef-43b9-b68d-025b827a3e51`
- Model version: `be5891fc-1440-482b-b5d5-5eddec45f958`

The same Grand Challenge image/model version identifiers appear in the August 30 preliminary evaluation. A verified mapping from these platform identifiers to a local Docker image digest/checkpoint has not been recovered. Consequently, this repository does not claim exact reproduction of the final score from a chosen local directory.

## Docker variants

`docker/midog` converts BatchNorm layers to GroupNorm. `docker/midog_v2` retains BatchNorm. Their `ASPP.py` and `DeepLabv3_plus.py` differ; their recovered inference driver, postprocessing helpers, entrypoint, dependency file and YAML configurations match. Keep each network with its appropriate weights.

Both build contexts expect weights at `/opt/ml/model/last.ckpt`, with a resources-directory fallback. They do not embed a checkpoint in the Docker image. The loader strips `net.` or `model.` prefixes and calls `load_state_dict(..., strict=False)`. This makes a successful load insufficient proof of full checkpoint compatibility; inspect missing/unexpected keys when validating a candidate.

The inference argument defaults take precedence over the YAML settings in the recovered driver. In particular, the driver defaults are tile size **1024**, stride **614**, peak minimum **0.1**, peak distance **18**, and mitotic threshold **0.5**; the saved patch YAML specifies **512/320**. `NMS_MM` defaults to **0.015**. The `candidate-mode` argument is parsed but not passed to the point-extraction call. These original behaviors are preserved, not silently corrected. Explicit environment overrides should be recorded with any new run.

`inference.py` processes the first supported image found. Grand Challenge calls the algorithm for an input case; this script is not a multi-image batch runner.

## Training snapshots

- `training/experiment_workspace/aug_logs/default/version_15/associated_codes` and `version_16/associated_codes` contain run-associated source. Corresponding saved network variants are under `network/v_15 train codes` and `network/v_16 train codes`.
- The top-level experiment `Train.py` was subsequently modified: it selects GPU index 2, sets a one-epoch continuation and references an absolute checkpoint path. It must not be described as the untouched 100-epoch submitted training recipe.
- `training/preprocessing_workspace` preserves patch/mask construction, stain augmentation and earlier training utilities.
- `docker/midog_v2/training_resources` preserves the unpacked training bundle found beside that submission.
- Old and `different_version` subdirectories are historical alternatives, not an instruction to run all variants.

Paths, splits, normalization, architecture, weights and thresholds must be reconciled with a particular saved run before scientific replication. No training, retuning or test-set evaluation was performed as part of organization.

## Environment

The Dockerfiles pin the historical CUDA 11.1 / Torch 1.9 stack. The owner's current working environment contains Python 3.10.11, Torch 2.7.1+cu118, Lightning 2.5.6 and NumPy 2.2.6; `lightning-bolts` is absent. It is appropriate for the dependency-free recovery checks, but it is not the original inference environment. The environment was left unchanged.

Validation covers preserved-file hashes, syntax, JSON, shell syntax, required Docker COPY sources and publication exclusions. Historical images were inventoried and their source recovered. The current build-context source matches the corresponding installed `midogv1_deeplab` and `midogv2_deeplab` images file for file. Fresh image builds, checkpoint loading and full model execution remain unverified.
