# Attribution and source status

This repository preserves the owner's recovered challenge code and the reference copies present alongside it. The owner initialized the GitHub repository with the [Apache 2.0 license](../LICENSE), which is retained unchanged. Original comments and attributions remain in the recovered files. Third-party components retain their applicable upstream terms.

Relevant upstream implementations and dependencies:

- [MIDOG 2025 Track 1 reference container](https://github.com/DeepMicroscopy/MIDOG25_T1_reference_docker): recovered reference directories under `evaluation` and `training/preprocessing_workspace/tensor_csv`. Some local files have been modified; a reference directory name does not establish that it is an untouched upstream checkout.
- [MIDOG 2025 Track 1 evaluation container](https://github.com/DeepMicroscopy/MIDOG25_T1_evaluation_docker): recovered evaluation helpers.
- [PyTorch Lightning Bolts](https://github.com/Lightning-Universe/lightning-bolts): ResNet implementation imported by the network. Training docstrings credit Annika Brundyn's Lightning segmentation example.
- [RandStainNA](https://github.com/yiqings/RandStainNA): referenced by the recovered stain-augmentation source.
- Recovered normalization utilities credit Python for Microscopists and refer to Macenko stain normalization and its linked implementations.

Dataset licenses and access conditions remain with their original providers. No dataset images, annotation databases, private per-image evaluation exports or downloaded model weights are included in this repository. The contribution paper is cited by DOI; the local PDF is retained in the local collection.
