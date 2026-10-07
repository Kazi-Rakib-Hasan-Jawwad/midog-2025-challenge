"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

from torchvision.transforms import Lambda

import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from torch.nn.functional import one_hot
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def extract_predictions(probabilities, confidence_threshold):
    indices = np.meshgrid(np.arange(0, probabilities.shape[1]), np.arange(0, probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    indices_y = indices_y.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    probabilities = probabilities.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    boxes_pred = np.concatenate((indices_x, indices_y, probabilities), axis=1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:, 2] >= confidence_threshold, :]
    return boxes_pred


def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i, :2], 0), points[:, :2])
            hit = np.argwhere(hit <= distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i, :])
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
    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 30
    initial_threshold = 0.1

    predicted_detections1 = []
    ground_truth_detections1 = []
    predicted_detections2 = []
    ground_truth_detections2 = []

    preds = torch.sigmoid(input)

    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels)
    labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels)
    labels_11 = labels_1.detach().cpu().numpy()
    labels_22 = labels_2.detach().cpu().numpy()

    preds_1 = Lambda(lambda x: x[:, 1, :, :])(preds)
    preds_2 = Lambda(lambda x: x[:, 2, :, :])(preds)
    # preds_00 = preds_0.detach().float().cpu().numpy().astype(np.float32, copy=False)
    preds_11 = preds_1.detach().float().cpu().numpy().astype(np.float32, copy=False)
    preds_22 = preds_2.detach().float().cpu().numpy().astype(np.float32, copy=False)

    # autocast-safe entropy + error weights
    with torch.no_grad():
        # work in fp32 for logs, then cast back
        probs_fp32 = torch.sigmoid(input.detach().to(torch.float32))  # (B,C,H,W)
        entropy = -(probs_fp32 * torch.log(probs_fp32.clamp_min(1e-8)) +
                    (1.0 - probs_fp32) * torch.log((1.0 - probs_fp32).clamp_min(1e-8)))
        # penalize confident mistakes too
        error_term = torch.abs(labels.to(torch.float32) - probs_fp32)
        # combine and bring back to model dtype (fp16 under AMP)
        weight = (1.0 + entropy + error_term).to(input.dtype)

    # 1. Calculate per-pixel loss. Note the reduction='none'.
    bce_pixelwise = F.binary_cross_entropy_with_logits(input, labels, reduction='none')

    # 2. Apply the per-pixel weights. This is an element-wise multiplication.
    weighted_loss_pixelwise = bce_pixelwise * weight

    # 3. Now, reduce the final weighted loss tensor to a single scalar value.
    bce_loss = weighted_loss_pixelwise.mean()

    for i in range(len(preds_11)):
        # print(i)
        pred_temp1 = extract_predictions(preds_11[i], confidence_threshold=initial_threshold)
        predicted_detections1.append(non_max_supression_distance(pred_temp1, distance_threshold=50))
    del pred_temp1

    for i in range(len(preds_22)):
        # print(i)
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
