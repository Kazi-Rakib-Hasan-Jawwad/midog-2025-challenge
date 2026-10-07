# Challenge results

## Final submission phase

Source: [official leaderboard](https://midog2025.grand-challenge.org/evaluation/track-1-mitosis-detection-final-submission-phase/leaderboard/) and [evaluation](https://midog2025.grand-challenge.org/evaluation/6f3f6c11-3ae0-47ec-99f6-eafed0bf45b1/), checked **7 October 2026**. Values below are the displayed rounded leaderboard values.

| Field | Value |
| --- | --- |
| # | 14th |
| User (Team) | krhasan02 (C-Omics) |
| Algorithm | DeepLabV3+FDA Aug |
| Created | 1 Sept. 2025 |
| F1 | 0.6336 |
| Precision | 0.7228 |
| Recall | 0.5639 |
| AP | 0.4869 |
| FROC AUC | 4.2434 |
| F1 (Hotspot ROIs) | 0.6627 |
| F1 (Random ROIs) | 0.6074 |
| F1 (Challenging ROIs) | 0.4518 |
| F1 (Tumor 1) | 0.7287 |
| F1 (Tumor 2) | 0.4524 |
| F1 (Tumor 3) | 0.7260 |
| F1 (Tumor 4) | 0.5458 |
| F1 (Tumor 5) | 0.5714 |
| F1 (Tumor 6) | 0.6847 |
| F1 (Tumor 7) | 0.6832 |
| F1 (Tumor 8) | 0.7265 |
| F1 (Tumor 9) | 0.5998 |
| F1 (Tumor 10) | 0.6164 |
| F1 (Tumor 11) | 0.2800 |
| F1 (Tumor 12) | 0.2041 |

There were **15 published leaderboard entries** at retrieval. The algorithm belongs to **krhasan02 (C-Omics)**.

## Preliminary evaluation records

These aggregate values were extracted from locally saved Grand Challenge evaluation PDFs. The full PDFs include per-image annotations and remain local.

| Algorithm | Evaluation | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| DeepLabV3+FDA Aug | `0ca7b452-fcee-4a16-a9b8-7f4cc7e76300` | 0.654490 | 0.723906 | 0.597222 |
| Deeplab v3 FDA 512px | `7fbd3752-e8b0-4fb6-b868-668c1ec22df7` | 0.579394 | 0.513978 | 0.663889 |

The contribution paper reports the preliminary **0.654** F1 for DeepLabV3+FDA Aug. The saved August 30 preliminary export shows position 21 at its capture; that historical position is not the final leaderboard rank. No preliminary-to-final score substitution is made.

Machine-readable records: [`final_results.json`](final_results.json), [`preliminary_results.json`](preliminary_results.json). The latter preserves all recovered aggregate fields, including organizer sentinel values such as AP = -1 for absent ROI categories; those are not additional valid performance estimates.
