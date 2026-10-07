
![MIDOG 2025 Logo](https://github.com/user-attachments/assets/6e28b009-7dcb-4bc1-b463-8528e943f952)

## MIDOG 2025 - Evaluation container for Track 1 (Mitotic Figure Detection)

This repository contains the official evaluation container used for the MIDOG 2025 MICCAI challenge. It is based on the starter kit provided by grand-challenge.org, and adds the relevant metrics.

Please note that obviously the ground-truth in this docker container is not part of the real test sets to be used in the challenge.

### How it works:

The ground truth and the predictions are fed to this container. The predictions are fed as the file `predictions.json`, but for each predicted image, an independent json file is provided.

The ground truth format is as follows:
```
{
 "007.tiff": 
   { "annotations": [[0.374850364, 0.756961752, 0 ]], 
     "tumor domain": 0, 
     "roi type": "hotspot"} ,
 "140.tiff": {
    "annotations": [[0.6732737644102278, 0.6380649407634927, 0], [0.701136862260162, 0.5339582751605569, 0], [0.004306115122262555, 0.017097810044277795, 0]],
    "tumor domain": 0,
    "roi type": "challenging"
 },
 "201.tiff":
    {"annotations": [[0.6596242342527185, 0.6275941892514884, 0],
 [0.6869224531269846, 0.5251959312989827, 0],
 [0.004218815644204747, 0.016817232145484504, 0]],
     "tumor domain": 1,
     "roi type": "random"
     }
}

```

where roi_type can be either `hotspot`, `challenging`, or `random`, according to the source of the image tile. In MIDOG 2025, we have three distinct dataset subsets:
- The hotspot ROIs, selected by an experienced pathologist in a cellularly dense tumor area
- The challenging images, which were taken from image parts that were deemed challenging for models due to having vast amounts of hard negatives (e.g., inflammatory tissue, necrotic tissue).
- The random images, selected at random from the tissue-filled area of the WSIs, statistically representing the WSI.

The mitotic-figure.json files to be expected from the docker containers of the participants have the following format:
```
{
    "name": "Points of interest",
    "type": "Multiple points",
    "points": [
        {
            "name": "mitotic figure",
            "point": [
                0.04,
                0.8,
                0
            ],
            "probability": 0.92
        },
        {
            "name": "non-mitotic figure",
            "point": [
                1.3,
                1.1,
                0
            ],
            "probability": 0.4
        }
    ],
    "version": {
        "major": 1,
        "minor": 0
    }
}
```

Here, each `point` represents a detection, given by its x, y and z coordinate. We expect the z coordinate to be always 0. ***Note that the coordinates are given in millimeters, not pixels.*** 
The example docker containers provided by us [here](https://github.com/DeepMicroscopy/MIDOG25_T1_reference_docker) perform a translation between pixels and millimeters for each input image.

Please note that the name of each point needs to be either `mitotic figure` or `non-mitotic figure`, depending on if the model decides it to be above threshold or not. This represents the classification information in the detection task.
Independent of this decision, we require the probability of the model for each cell, i.e., the confidence of the model for each cell. This is used for calculation of metrics such as average precision.

