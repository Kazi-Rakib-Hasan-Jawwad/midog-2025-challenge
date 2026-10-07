"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

import torch
import functools
#from ND_Crossentropy import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss
from torch import nn
from torchvision.transforms import Lambda
from PIL import Image
import cv2
from torch.autograd import Variable
from torch import einsum

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label, center_of_mass
from sklearn.metrics import f1_score
from torch.nn.functional import one_hot


from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

'''
def extract_predictions(probabilities, confidence_threshold):
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]
    return boxes_pred


def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)
    return wanted

def count_tp_fp_fn(pred, gt, prob_threshold, hit_distance):
    # This function counts the number of true-positives, false negatives and false positives in a detection taks.

    # inputs: A numpy array of shape [N,3] for pred and gt. N is the number of detections. The first column is the x-position of the detections.
    # the second column is the y-position of the detections. the third column is the probability associated with detections.
    # The hit-distance is the maximum distance of a detection from the ground-truth to be counted as a true-positive, otherwsie, it will be counted as a false positive.
    # The prob_threshld is the threshold value in which detections with probabilities smaller than the threshold value will be discarded.

    # output: number of ture-positives, false positives, and false negatives.
    if pred.shape[0] > 0:
        pred = pred[np.argwhere(pred[:, 2] >= prob_threshold)[:, 0], :2]
        gt = gt[np.argwhere(gt[:, 2] >= prob_threshold)[:, 0], :2]
    if pred.shape[0] > 0:
        if gt.shape[0] > 0:
            hit = cdist(pred[:, :2], gt)
            hit[hit <= hit_distance] = 1
            hit[hit != 1] = 0
            sum_1 = np.sum(hit, 0)
            sum_2 = np.sum(hit, 1)
            fp = sum(sum_2 == 0)
            fn = sum(sum_1 == 0)
            tp = sum(sum_1 != 0)
        else:
            tp = 0
            fn = 0
            fp = pred.shape[0]
    else:
        tp = 0
        fp = 0
        fn = gt.shape[0]
    return tp, fp, fn


def evaluate_with_multiple_thresholds(pred, gt, thresholds, hit_distance):
    results = []
    for th in thresholds:
        tp, fp, fn = count_tp_fp_fn(pred, gt, prob_threshold=th, hit_distance=hit_distance)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            'threshold': th,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    return results


######## Calculating TP,FP,FN for each Patch, Add them all and then calculate Sen,FP-Per-Patch

def normalize_detections(detections_list):
    normalized = []
    for det in detections_list:
        det = np.array(det)
        if det.size == 0:
            continue  # skip empty arrays

        if det.ndim == 1:
            if det.shape[0] == 3:
                det = det[np.newaxis, :]  # (3,) → (1, 3)
            else:
                continue  # skip invalid 1D entries
        elif det.ndim == 2:
            if det.shape[1] != 3:
                continue  # skip if not 3 columns
        else:
            continue  # invalid shape

        normalized.append(det)
    return normalized


def Class_Wise_TIL_Detection_FROC(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01

    confidence_thresholds = np.linspace(0.1, 0.9, 40)
    distance_threshold = 30
    predicted_detections1 = []
    ground_truth_detections1 = []
    predicted_detections2 = []
    ground_truth_detections2 = []

    preds = F.softmax(input)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels) # background class of targets
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels) # mitotic figure
    labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels) # non mitotic figure
    labels_00 = labels_0.detach().cpu().numpy()
    labels_11 = labels_1.detach().cpu().numpy()
    labels_22 = labels_2.detach().cpu().numpy()


    preds_0 = Lambda(lambda x: x[:, 0, :, :])(preds) #background class of predictions
    preds_1 = Lambda(lambda x: x[:, 1, :, :])(preds) # mitotic figure
    preds_2 = Lambda(lambda x: x[:, 2, :, :])(preds) # non mitotic figure
    preds_00 = preds_0.detach().cpu().numpy()
    preds_11 = preds_1.detach().cpu().numpy()
    preds_22 = preds_2.detach().cpu().numpy()

    # get intersection
    intersection_0 = (preds_0 * labels_0).sum()
    intersection_1 = (preds_1 * labels_1).sum()

    intersection_SUM = intersection_0 + intersection_1

    # get union
    union_0 = (preds_0.sum() + labels_0.sum()) + eps
    union_1 = (preds_1.sum() + labels_1.sum()) + eps
    union_SUM = union_0 + union_1

    # get dice total score and dice loss
    dice_total = (2 * intersection_SUM / union_SUM)
    dice_total_loss = 1 - dice_total

    dice0 = (2 * intersection_0 / union_0)
    dice1 = (2 * intersection_1 / union_1)


    bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')


    for i in range(len(preds_11)):
        #print(i)
        pred_temp1 = extract_predictions(preds_11[i], confidence_threshold=0.1)
        predicted_detections1.append(non_max_supression_distance(pred_temp1, distance_threshold=50))
    del pred_temp1

    for i in range(len(preds_22)):
        #print(i)
        pred_temp2 = extract_predictions(preds_22[i], confidence_threshold=0.1)
        print(preds_22.shape)
        predicted_detections2.append(non_max_supression_distance(pred_temp2, distance_threshold=50))
    del pred_temp2

    for i in range(len(labels_11)):
        #print(i)
        labels_temp1 = extract_predictions(labels_11[i], confidence_threshold=0.1)
        ground_truth_detections1.append(non_max_supression_distance(labels_temp1, distance_threshold=50))
    del labels_temp1

    for i in range(len(labels_22)):
        labels_temp2 = extract_predictions(labels_22[i], confidence_threshold=0.1)
        ground_truth_detections2.append(non_max_supression_distance(labels_temp2, distance_threshold=50))

    del labels_temp2

    ground_truth_detections1 = normalize_detections(ground_truth_detections1)
    predicted_detections1 = normalize_detections(predicted_detections1)

    ground_truth_detections2 = normalize_detections(ground_truth_detections2)
    predicted_detections2 = normalize_detections(predicted_detections2)

    pred_all_1 = np.concatenate(predicted_detections1, axis=0) if len(predicted_detections1) > 0 else np.empty((0, 3))
    gt_all_1 = np.concatenate(ground_truth_detections1, axis=0) if len(ground_truth_detections1) > 0 else np.empty(
        (0, 3))

    pred_all_2 = np.concatenate(predicted_detections2, axis=0) if len(predicted_detections2) > 0 else np.empty((0, 3))
    gt_all_2 = np.concatenate(ground_truth_detections2, axis=0) if len(ground_truth_detections2) > 0 else np.empty(
        (0, 3))

    results_1 = evaluate_with_multiple_thresholds(pred_all_1, gt_all_1, confidence_thresholds, 30)
    results_2 = evaluate_with_multiple_thresholds(pred_all_2, gt_all_2, confidence_thresholds, 30)
    best_result_1 = max(results_1, key=lambda x: x['f1'])
    best_result_2 = max(results_2, key=lambda x: x['f1'])
    f1_score_1 = best_result_1['f1']
    f1_score_2 = best_result_2['f1']


    return bce_loss, f1_score_1, f1_score_2
'''


def extract_predictions(probabilities, confidence_threshold):
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]
    return boxes_pred


def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)
    return wanted


def extract_true_centers(prob_map, confidence_threshold=0.5, min_distance=10):
    """
    확률 맵에서 confidence_threshold 이상 픽셀을 바이너리 마스크로 변환 후,
    Distance Transform과 Watershed를 이용해 연결된 오브젝트 분리 및
    각 오브젝트 내에서 거리 변환 값이 가장 큰 픽셀(진짜 중심)만 추출

    Args:
        prob_map (2D np.ndarray): 확률 맵 (height x width)
        confidence_threshold (float): 확률 임계값
        min_distance (int): local maxima 탐색 시 최소 거리 (픽셀 단위)

    Returns:
        centers (np.ndarray): (N, 2) 형태 배열, 각 행은 [x, y] 좌표
    """

    # 1. Thresholding → binary mask
    binary_mask = prob_map >= confidence_threshold
    if np.sum(binary_mask) == 0:
        return np.zeros((0, 2))  # 검출된 영역 없음

    # 2. Distance transform (각 픽셀에서 가장 가까운 배경까지 거리)
    distance = ndi.distance_transform_edt(binary_mask)

    # 3. Local maxima (중심 후보) 탐색 (min_distance 간격으로)
    local_maxi = peak_local_max(distance, indices=False, min_distance=min_distance, labels=binary_mask)

    # 4. Marker labeling (local maxima 각각에 레이블 부여)
    markers, num_features = ndi.label(local_maxi)

    # 5. Watershed segmentation (markers 기준 오브젝트 분리)
    labels = watershed(-distance, markers, mask=binary_mask)

    centers = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labels == i)
        if coords.size == 0:
            continue
        # 6. 각 오브젝트 내부에서 distance transform 값이 최대인 픽셀을 중심으로 선택
        d_vals = distance[labels == i]
        max_idx = np.argmax(d_vals)
        yx = coords[max_idx]  # (y, x)
        centers.append([yx[1], yx[0]])  # [x, y] 형태로 저장

    return np.array(centers)

def count_tp_fp_fn(pred, gt, prob_threshold, hit_distance):
    pred = np.array(pred)
    gt = np.array(gt)

    if pred.shape[0] > 0:
        pred = pred[np.argwhere(pred[:, 2] >= prob_threshold)[:, 0], :2]
    else:
        pred = np.zeros((0, 2))

    # ✅ 방어코드 추가: gt가 2차원이 아닐 경우 처리
    if gt.ndim != 2 or gt.shape[0] == 0:
        gt = np.zeros((0, 2))  # 빈 GT로 간주
    elif gt.shape[1] == 3:
        gt = gt[np.argwhere(gt[:, 2] >= prob_threshold)[:, 0], :2]
    else:
        gt = gt[:, :2]

    if pred.shape[0] > 0:
        if gt.shape[0] > 0:
            hit = cdist(pred[:, :2], gt)
            hit[hit <= hit_distance] = 1
            hit[hit != 1] = 0
            sum_1 = np.sum(hit, 0)
            sum_2 = np.sum(hit, 1)
            fp = sum(sum_2 == 0)
            fn = sum(sum_1 == 0)
            tp = sum(sum_1 != 0)
        else:
            tp = 0
            fn = 0
            fp = pred.shape[0]
    else:
        tp = 0
        fp = 0
        fn = gt.shape[0]

    return tp, fp, fn


######## Calculating TP,FP,FN for each Patch, Add them all and then calculate Sen,FP-Per-Patch

def compute_best_f1(pred, gt, confidence_thresholds, hit_distance):
    max_f1 = 0
    best_threshold = 0

    # ✅ 가장 짧은 리스트 기준으로 loop 돌리기
    num_samples = min(len(pred), len(gt))

    for threshold in confidence_thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        for N in range(num_samples):
            tp, fp, fn = count_tp_fp_fn(pred[N], gt[N], threshold, hit_distance)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        if total_tp + total_fp + total_fn == 0:
            continue
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold

    return max_f1



def Class_Wise_TIL_Detection_FROC(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01

    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 30
    initial_threshold = 0.1

    predicted_detections1 = []
    ground_truth_detections1 = []
    predicted_detections2 = []
    ground_truth_detections2 = []

    preds = F.softmax(input, dim=1)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels)
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels)
    labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels)
    labels_00 = labels_0.detach().cpu().numpy()
    labels_11 = labels_1.detach().cpu().numpy()
    labels_22 = labels_2.detach().cpu().numpy()

    preds_0 = Lambda(lambda x: x[:, 0, :, :])(preds)
    preds_1 = Lambda(lambda x: x[:, 1, :, :])(preds)
    preds_2 = Lambda(lambda x: x[:, 2, :, :])(preds)
    preds_00 = preds_0.detach().cpu().numpy()
    preds_11 = preds_1.detach().cpu().numpy()
    preds_22 = preds_2.detach().cpu().numpy()


    # logits: input, targets: labels (B,3,H,W) from one_hot_label
    with torch.no_grad():
        valid_mask = (labels.sum(dim=1, keepdim=True) > 0).float()  # 1 where not ignored

    bce = torch.nn.functional.binary_cross_entropy_with_logits(input, labels, reduction='none')
    bce_loss = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    for i in range(len(preds_11)):
        #print(i)
        pred_temp1 = extract_predictions(preds_11[i], confidence_threshold=initial_threshold)
        predicted_detections1.append(non_max_supression_distance(pred_temp1, distance_threshold=50))
    del pred_temp1

    for i in range(len(preds_22)):
        #print(i)
        pred_temp2 = extract_predictions(preds_22[i], confidence_threshold=initial_threshold)
        predicted_detections2.append(non_max_supression_distance(pred_temp2, distance_threshold=50))
    del pred_temp2

    for i in range(len(labels_11)):
        label_temp1 = extract_true_centers(labels_11[i], confidence_threshold=initial_threshold,
                                                       min_distance=10)
        ground_truth_detections1.append(label_temp1)

    for i in range(len(labels_22)):
        label_temp2 = extract_true_centers(labels_22[i], confidence_threshold=initial_threshold,
                                                       min_distance=10)
        ground_truth_detections2.append(label_temp2)

    f1_1 = compute_best_f1(predicted_detections1, ground_truth_detections1, confidence_thresholds, distance_threshold)
    f1_2 = compute_best_f1(predicted_detections2, ground_truth_detections2, confidence_thresholds, distance_threshold)

    return bce_loss, f1_1, f1_2
