# Source provenance

The local originals were copied on 7 October 2026. Publication changes add documentation, checksum records, ignore rules and a validation helper; they do not rewrite the recovered implementation.

| Source ID | Original project-relative origin | Published location |
| --- | --- | --- |
| `docker` | `MIDOG/docker_files` | `docker` |
| `training_experiment_workspace` | `MIDOG/MIDOG_2025_MICCAI` | `training/experiment_workspace` |
| `training_preprocessing_workspace` | `MIDOG2025_MICCAI` | `training/preprocessing_workspace` |
| `early_challenge` | `MIDOG2025_Challenge` | `archive/early_challenge` |
| `evaluation_workspace` | `MIDOG2025_Eval` | `evaluation` |
| `desktop_experiments` | Selected MIDOG files in the desktop experiment folder | `archive/desktop_experiments` |
| `docker_image:sha256:…` | `/opt/algorithm` or image entrypoint, identified by full image digest | `archive/docker_images/<tag>` |

The machine-readable [`source_manifest.json`](source_manifest.json) records every recovered file. Local absolute paths, private evaluation exports and data inventories remain in the separate local collection. The manifest is regenerated only after comparing copied source with its original or the recorded image extraction hash.

All **14 extracted files** from `midogv1_deeplab:latest` match the corresponding `docker/midog` files, and all **14 extracted files** from `midogv2_deeplab:latest` match `docker/midog_v2`. These are verified matches between recovered source and local images; they do not map those images to the Grand Challenge image-version UUID. See [`docker_source_comparison.json`](docker_source_comparison.json) for the full comparison.

The earlier `midog_deeplab_v2` and `midog_deeplab_v3` images contain different inference/postprocessing/configuration sources, which are retained as separate historical snapshots. The oldest image, `midog-alg-deeplab_v1` (also tagged `midog-deeplab_v2`), lacks `/opt/algorithm/inference.py`; its source snapshot is incomplete and should not be presented as a working build.

The original `training_resources.zip` was compared with its unpacked directory: all 34 non-cache files match. Its unpacked source/configuration is retained under `docker/midog_v2/training_resources`; generated cache/data artifacts remain local.

The mounted training-project copy and later MIDOG research projects are cataloged locally. They have not been conflated with the 2025 submission or its final leaderboard result.

The desktop fragments were found by searching file contents for MIDOG after the folder-name search. Their four CSV files contain only `Wall time`, `Step` and `Value` scalar columns. Their exact training-run/checkpoint association was not recovered; they are retained as unbound historical curves, not additional challenge metrics. The remaining content-search matches outside MIDOG roots were TIGER inventory utilities referring to MIDOG, and were excluded as unrelated source.
