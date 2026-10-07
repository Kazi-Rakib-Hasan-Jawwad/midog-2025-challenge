"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json


import random
from statistics import mean
from pathlib import Path
from torch import Tensor, IntTensor
from pprint import pformat, pprint
import numpy as np
from helpers import run_prediction_processing, tree
from evalutils.scorers import score_detection 

from torchmetrics.detection.mean_ap import MeanAveragePrecision


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

MAXIMUM_MITOSIS_DETECTION_DISTANCE=7.5e-3
roi_types = ["hotspot","challenging","random"]

def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any specific order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently

    metrics["results"] = []
    for result in predictions: 
        print(f"Processing job {result['pk']}...")
        # Each job is processed in a separate process
        metrics["results"].append(process(result))

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission

    tumordomains = np.unique([result["tumor_domain"] for result in metrics["results"]])

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', max_detection_thresholds=[1,10,1e6], rec_thresholds=np.arange(0,1.01,0.01).tolist())
    per_tumor_map_metric = {d : MeanAveragePrecision(box_format='xyxy', iou_type='bbox', max_detection_thresholds=[1,10,1e6], rec_thresholds=np.arange(0,1.01,0.01).tolist()) for d in tumordomains}
    per_tumor = {d : {'tp': 0, 'fp':0, 'fn':0} for d in tumordomains}
    per_roi_type = {d : {'tp': 0, 'fp':0, 'fn':0} for d in roi_types}
    per_roi_type_map_metric = {d : MeanAveragePrecision(box_format='xyxy', iou_type='bbox', max_detection_thresholds=[1,10,1e6], rec_thresholds=np.arange(0,1.01,0.01).tolist()) for d in roi_types}


    tp,fp,fn = 0,0,0
    bbox_size = 0.01125 # equals to 7.5mm distance for horizontal distance at 0.5 IOU

    all_preds = []
    all_gts = []
    per_tumor_preds = {d: [] for d in tumordomains}
    per_tumor_gts = {d: [] for d in tumordomains}
    per_roi_type_preds = {d: [] for d in roi_types}
    per_roi_type_gts = {d: [] for d in roi_types}

    for result in metrics["results"]:
        tp += result["metrics"]["true_positives"]            
        fp += result["metrics"]["false_positives"]            
        fn += result["metrics"]["false_negatives"] 

        pred_dict = [{'boxes': Tensor([[x-bbox_size,y-bbox_size, x+bbox_size, y+bbox_size] for (x,y,z,_,_) in result["pred"]]), 
                        'labels': IntTensor([1,]*len(result["pred"])),
                        'scores': Tensor([sc for (x,y,z,_,sc) in result["pred"]])}]

        all_preds.append(result["pred"])
        all_gts.append(result["gt"])
        target_dict = [{'boxes': Tensor([[x-bbox_size,y-bbox_size, x+bbox_size, y+bbox_size] for (x,y,z) in result["gt"]]),
                                'labels' : IntTensor([1,]*len(result["gt"]))}]

        map_metric.update(pred_dict,target_dict)
        per_tumor_map_metric[result["tumor_domain"]].update(pred_dict,target_dict)
        per_roi_type_map_metric[result["roi_type"]].update(pred_dict, target_dict)

        # accumulate for ROI type
        per_tumor[result["tumor_domain"]]['tp'] += result["metrics"]["true_positives"] 
        per_tumor[result["tumor_domain"]]['fp'] += result["metrics"]["false_positives"] 
        per_tumor[result["tumor_domain"]]['fn'] += result["metrics"]["false_negatives"] 

        per_tumor_preds[result["tumor_domain"]].append(result["pred"])
        per_tumor_gts[result["tumor_domain"]].append(result["gt"])  

        # accumulate for ROI type
        per_roi_type[result["roi_type"]]['tp'] += result["metrics"]["true_positives"] 
        per_roi_type[result["roi_type"]]['fp'] += result["metrics"]["false_positives"] 
        per_roi_type[result["roi_type"]]['fn'] += result["metrics"]["false_negatives"] 

        per_roi_type_preds[result["roi_type"]].append(result["pred"])
        per_roi_type_gts[result["roi_type"]].append(result["gt"])

    eps = 1E-6
    aggregate_results=dict()
    aggregate_results["precision"] = tp / (tp + fp + eps)
    aggregate_results["recall"] = tp / (tp + fn + eps)
    aggregate_results["f1_score"] = (2 * tp + eps) / ((2 * tp) + fp + fn + eps)
    aggregate_results["froc_auc"] = calc_froc_score(all_gts, all_preds) 

    metrics_values = map_metric.compute()
    aggregate_results["AP"] = metrics_values['map_50'].tolist()

    for tumor in per_tumor:
        aggregate_results[f'tumor_{tumor}_precision'] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fp'] + eps)
        aggregate_results[f'tumor_{tumor}_recall'] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fn'] + eps)
        aggregate_results[f'tumor_{tumor}_f1'] = (2 * per_tumor[tumor]['tp'] + eps) / ((2 * per_tumor[tumor]['tp']) + per_tumor[tumor]['fp'] + per_tumor[tumor]['fn'] + eps) 

        pt_metrics_values = per_tumor_map_metric[tumor].compute()
        aggregate_results[f"tumor_{tumor}_AP"] = pt_metrics_values['map_50'].tolist()
        aggregate_results[f'tumor_{tumor}_froc_auc'] = calc_froc_score(per_tumor_gts[tumor], per_tumor_preds[tumor])

    for roi_type in per_roi_type:
        aggregate_results[f'roi_type_{roi_type}_precision'] = per_roi_type[roi_type]['tp'] / (per_roi_type[roi_type]['tp'] + per_roi_type[roi_type]['fp'] + eps)
        aggregate_results[f'roi_type_{roi_type}_recall'] = per_roi_type[roi_type]['tp'] / (per_roi_type[roi_type]['tp'] + per_roi_type[roi_type]['fn'] + eps)
        aggregate_results[f'roi_type_{roi_type}_f1'] = (2 * per_roi_type[roi_type]['tp'] + eps) / ((2 * per_roi_type[roi_type]['tp']) + per_roi_type[roi_type]['fp'] + per_roi_type[roi_type]['fn'] + eps) 

        pt_metrics_values = per_roi_type_map_metric[roi_type].compute()
        aggregate_results[f"roi_type_{roi_type}_AP"] = pt_metrics_values['map_50'].tolist()

        aggregate_results[f'roi_type_{roi_type}_froc_auc'] = calc_froc_score(per_roi_type_gts[roi_type], per_roi_type_preds[roi_type])

    if metrics["results"]:
        metrics["aggregates"] = aggregate_results

    for k,_ in enumerate(metrics["results"]):
        metrics["results"][k]["gt"] = []

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key(job)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("histopathology-region-of-interest-cropout",): process_interf0,
    }[interface_key]

    # Call the handler
    return handler(job)

def calc_froc_score(ground_truth:list, detections:list, max_fp:int = 8, nbr_of_thresholds:int = 40):
    """ Calculate the FROC (free-response operating characteristic) score. 
    ground_truth: The ground truth annotations for the images.
    detections: The detected mitotic figures.

    This metric calculates the trade-off between sensitivity (true positive rate) and the number of false positives per image at various detection thresholds.

    We use a maximum of 8 false positives per image for the calculation, as in: https://jamanetwork.com/journals/jama/fullarticle/2665774

    More on this metric: https://metrics-reloaded.dkfz.de/metric?id=FROC_score
    """
    if len(detections) == 0:
        return float('nan')
    
    valid_detections = [x for x in detections if len(x) > 0]
    if len(valid_detections) == 0:
        return float('nan')

    all_detections_in_all_images = np.vstack([x for x in detections if len(x) > 0])
    all_thresholds = np.array([sc for x,y,z,cls,sc in all_detections_in_all_images])
    if np.min(all_thresholds) == np.max(all_thresholds):
        threshold_list = [np.min(all_thresholds)]
    else:
        threshold_list = (np.linspace(np.min(all_thresholds),np.max(all_thresholds),nbr_of_thresholds)).tolist()

    recalls={}
    fppi={}
    tps={}
    fns={}
    fps_per_image={}
    for threshold in threshold_list:
        tps[threshold] = 0
        fns[threshold] = 0
        fps_per_image[threshold] = []
        for i in range(len(detections)):
            filtered_predictions = [[x,y,0] for x,y,z,cls,sc in detections[i] if sc>=threshold]
            sc = score_detection(ground_truth=ground_truth[i],predictions=filtered_predictions,radius=MAXIMUM_MITOSIS_DETECTION_DISTANCE)._asdict()
            fps_per_image[threshold].append(sc['false_positives'])
            tps[threshold] += sc['true_positives']
            fns[threshold] += sc['false_negatives']
        recalls[threshold] = tps[threshold] / (tps[threshold] + fns[threshold] + 1E-6)
        fppi[threshold] = np.mean(fps_per_image[threshold])

    # Sort by FPPI
    fppi_arr = np.array(list(fppi.values()))
    recall_arr = np.array(list(recalls.values()))
    sort_idx = np.argsort(fppi_arr)
    fppi_sorted = fppi_arr[sort_idx]
    recall_sorted = recall_arr[sort_idx]

    fppi_eval = np.linspace(0, max_fp, 50)
    recall_interp = np.interp(fppi_eval, fppi_sorted, recall_sorted, left=0, right=recall_sorted[-1])

    froc_auc = np.trapz(recall_interp, fppi_eval)

    return froc_auc

def process_interf0(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_mitotic_figures = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="mitotic-figures",
    )

    # Secondly, read the results
    result_mitotic_figures = load_json_file(
        location=location_mitotic_figures,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth
    image_name_histopathology_region_of_interest_cropout = get_image_name(
        values=job["inputs"],
        slug="histopathology-region-of-interest-cropout",
    )

    # Fourthly, load your ground truth

    ground_truth_dir = Path("/opt/ml/input/data/ground_truth")
    with open(
        ground_truth_dir / "ground_truth.json", "r"
    ) as f:
        truth = json.loads(f.read())

    if image_name_histopathology_region_of_interest_cropout not in truth:
        raise NameError(f'Ground truth for processed image {image_name_histopathology_region_of_interest_cropout} not found.')
    
    if 'points' not in result_mitotic_figures:
        raise SyntaxError('Results must contain dictionary with "points" field. ')

    if 'annotations' not in truth[image_name_histopathology_region_of_interest_cropout]:
        raise SyntaxError(f"No GT annotations found case {image_name_histopathology_region_of_interest_cropout}.")
    
    if not isinstance(truth[image_name_histopathology_region_of_interest_cropout], dict):
        raise TypeError(f"Annotation for {image_name_histopathology_region_of_interest_cropout} is not of type dictionary. ")
        
    if 'roi type' not in truth[image_name_histopathology_region_of_interest_cropout]:
        raise ValueError(f"Value 'roi type' missing for GT annotation for image {image_name_histopathology_region_of_interest_cropout}")
    
    if truth[image_name_histopathology_region_of_interest_cropout]['roi type'] not in roi_types:
        raise ValueError(f'Invalid ROI type definition {truth[image_name_histopathology_region_of_interest_cropout]["roi type"]}. Valid values are: {str(roi_types)}')
        

    points=[]
    valid_names = ['mitotic figure','non-mitotic figure']
    for point in result_mitotic_figures['points']:
            
            if not isinstance(point, dict):
                raise SyntaxError('All points need to be dictionaries. Please see example output for required format. ')
            
            # New for MIDOG 2025: We no longer accept the MIDOG 21 output format, but require the MIDOG 22 output format.
            if 'name' not in point:
                raise SyntaxError('Field name is not part of detections. Please see example output for required format. ')

            if 'probability' not in point:
                raise SyntaxError('Field probability is not part of detections. Please see example output for required format. ')
            
            if 'point' not in point:
                raise SyntaxError('Point is not part of points structure. Please see example output for required format. ')
            
            if point['name'] not in valid_names:
                raise ValueError(f'Invalid setting for class of detection: {point["name"]}. Valid values are: {str(valid_names)}')
            
            detected_class = 1 if point['name']=='mitotic figure' else 0
            detected_thr   = point['probability']

            if point['point'][0] < -MAXIMUM_MITOSIS_DETECTION_DISTANCE or point['point'][1] < -MAXIMUM_MITOSIS_DETECTION_DISTANCE:
                raise ValueError(f"Invalid point coordinates for detection: {point['point']}")

            if point['point'][0] > 10 or point['point'][1] > 10:
                raise ValueError(f"Invalid point coordinates for detection: {point['point']}. Is it possible you used pixel coordinates in the output format? You need to provide the output coordinates in millimeters.")

            points.append([*point['point'][0:3], detected_class, detected_thr])


    # TODO: compare the results to your ground truth and compute some metrics
    filtered_predictions = [[x,y,0] for x,y,z,cls,sc in points if cls==1]

    bbox_size = 0.01125 # equals to 7.5mm distance for horizontal distance at 0.5 IOU
    
    sc = score_detection(ground_truth=truth[image_name_histopathology_region_of_interest_cropout]['annotations'],predictions=filtered_predictions,radius=MAXIMUM_MITOSIS_DETECTION_DISTANCE)._asdict()
    
 
    return {
        "image" : image_name_histopathology_region_of_interest_cropout,
        "gt" : truth[image_name_histopathology_region_of_interest_cropout]['annotations'],
        "tumor_domain" : truth[image_name_histopathology_region_of_interest_cropout]['tumor domain'],
        "pred" : points,
        "metrics" : sc,
        "roi_type" : truth[image_name_histopathology_region_of_interest_cropout]['roi type']
    }




def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    print("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        print(line)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    # Each interface has a unique key that is the set of socket slugs given as input
    socket_slugs = [sv["interface"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
