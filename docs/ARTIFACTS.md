# Model and container artifacts

Large artifacts are indexed locally and are not stored in Git. The local collection has shortcuts to original directories and a compressed file inventory; these shortcuts are references, not independent backups. Original files were not moved or deleted.

| Artifact in original `docker_files` | Bytes | Role |
| --- | ---: | --- |
| `midog/midogv1_deeplab.tar.gz` | 4,750,677,363 | Saved Docker image archive |
| `midog_v2/midogv2_deeplab2.tar.gz` | 4,750,696,413 | Saved alternate Docker image archive |
| `last.ckpt` | 485,100,091 | Recovered checkpoint candidate |
| `midog/resources/last.ckpt` | 485,100,091 | Byte-identical duplicate of the previous checkpoint |
| `model.tar.gz` | 443,427,415 | Saved model package |
| `midogv2_model/last.ckpt` | 485,611,935 | Alternate checkpoint candidate |
| `midogv2_model/model_v2.tar.gz` | 444,924,970 | Alternate model package |
| `midog_v2/training_resources.zip` | 1,001,192 | Original training-source package; unpacked source retained in Git |

Checkpoint SHA-256:

```text
45126f830f19b0ec109d9c17d204056095c62d0cd3d0ea10ba471807d68c8a2b  last.ckpt (also midog/resources/last.ckpt)
3ed093ad968022a892d881b3a876eddb858cbb31aa4891e0c2cac13e96273472  midogv2_model/last.ckpt
```

The archive files were inventoried by path and size; their full contents have not been equated to the running Docker image store. Image tags can change; use the digests in `docker_images.json` to identify the recovered local images. Checkpoint filenames alone do not identify a submitted model.

Additional epoch checkpoints, event logs, predictions, MIDOG++/CMC/CCMCT data, NAS archives and later research packages are listed in the **local** collection inventory. They are intentionally outside the public repository. Model redistribution can be handled separately after identifying the exact submitted weights and their intended release location.
