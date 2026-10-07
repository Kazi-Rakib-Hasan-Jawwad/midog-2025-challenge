# Source inventory

All recovered implementation files are listed in [`source_manifest.json`](source_manifest.json), with their original relative path, source identifier, byte count and SHA-256.

| Source group | Preserved files | Bytes |
| --- | ---: | ---: |
| desktop_experiments | 9 | 41,146 |
| docker_image | 69 | 130,357 |
| early_challenge | 13 | 52,355 |
| docker | 59 | 203,137 |
| evaluation_workspace | 67 | 215,357 |
| training_experiment_workspace | 67 | 324,567 |
| training_preprocessing_workspace | 67 | 282,229 |

**351 preserved files, 1,249,148 source bytes.** Documentation, validation tooling and ignore files were added during organization.

Source copies were checked against the originals. Identical files are retained where necessary to preserve separate historical build contexts or complete source snapshots. No existing workspace was moved, deleted or edited.

The separate local collection contains a full file-level inventory, original-path mappings, duplicate-source groups, archive member listings, recovered PDFs, later MIDOG research source and shortcuts to models/datasets. Environment, Git database, editor-cache and agent-state directories are recorded as excluded scopes; symbolic-link directories are recorded without recursively following them. Raw data, private evaluation exports, checkpoints and image archives are outside this public repository.
