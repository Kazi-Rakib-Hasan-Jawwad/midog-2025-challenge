"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

from torchvision.transforms import Lambda

import math, numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

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

# --- THE CORRECTED MAIN FUNCTION ---
def calculate_bce_loss_and_metrics(logits, one_hot_labels, calculate_metrics=False):
    """
    Calculates a hybrid loss (BCE + Mutual Exclusivity Penalty) and optionally computes detection metrics.
    """
    # --- 1. Calculate the standard BCE Loss ---
    # This part is the same as your original implementation.
    with torch.no_grad():
        valid_mask = (one_hot_labels.sum(dim=1, keepdim=True) > 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, one_hot_labels, reduction='none')

    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(1)  # (B,H,W)
        entropy = entropy / math.log(logits.shape[1])  # max‑norm
        weight = (1. + entropy).unsqueeze(1)

    bce = bce * valid_mask  # mask first
    weighted = bce * weight  # broadcast ok
    total_loss = weighted.sum() / valid_mask.sum().clamp_min(1)
    #bce_loss = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    '''
    # --- 2. NEW: Calculate the Mutual Exclusivity Penalty ---
    # We use probabilities (after sigmoid) for this part.
    probs = torch.sigmoid(logits)
    prob_background = probs[:, 0, :, :]
    prob_mitotic = probs[:, 1, :, :]
    prob_non_mitotic = probs[:, 2, :, :]

    # Penalize the model if it predicts high probabilities for multiple classes for the same pixel.
    penalty = (prob_mitotic * prob_non_mitotic) + \
              (prob_background * prob_mitotic) + \
              (prob_background * prob_non_mitotic)

    exclusivity_penalty = penalty.mean()

    # --- 3. Combine the two losses ---
    # The final loss is a combination of the BCE loss and the new penalty.
    total_loss = 0.6*bce_loss + 0.4 * exclusivity_penalty
    
    # --- 4. Metric Calculation (only for validation) ---
    if not calculate_metrics:
        return total_loss, 0.0, 0.0
    
    
    with torch.no_grad():
        confidence_thresholds = np.linspace(0.1, 0.9, 20)
        distance_threshold = 30

        probs_np = probs.cpu().numpy()
        labels_np = one_hot_labels.cpu().numpy()
    '''
    if not calculate_metrics:
        return total_loss, torch.tensor(0.), torch.tensor(0.)

    with torch.no_grad():
        logits_met, labels_met = logits, one_hot_labels.float()

        probs_np = torch.sigmoid(logits_met).cpu().numpy()
        labels_np = labels_met.cpu().numpy()

        probs_mitotic = probs_np[:, 1, :, :]
        probs_non_mitotic = probs_np[:, 2, :, :]

        gt_mitotic = labels_np[:, 1, :, :]
        gt_non_mitotic = labels_np[:, 2, :, :]

        f1_mitotic = compute_best_f1(probs_mitotic, gt_mitotic)
        f1_non_mitotic = compute_best_f1(probs_non_mitotic, gt_non_mitotic)

    return total_loss, f1_mitotic, f1_non_mitotic

# --- THE NEW MAIN FUNCTION WITH UNCERTAINTY WEIGHTING ---
def calculate_uncertainty_weighted_bce_loss_and_metrics(logits, one_hot_labels, calculate_metrics=False):
    """
    Calculates a hybrid loss where the BCE loss is dynamically weighted by the model's uncertainty.
    """
    # --- 1. Calculate the standard BCE Loss ---
    with torch.no_grad():
        valid_mask = (one_hot_labels.sum(dim=1, keepdim=True) > 0).float()
        bce_loss_pixelwise = F.binary_cross_entropy_with_logits(logits, one_hot_labels, reduction='none')

    # --- 2. NEW: Calculate Uncertainty (Entropy) ---
    # We use softmax probabilities to get a true probability distribution for entropy calculation.
    # This captures the model's confusion between the mutually exclusive classes.
    softmax_probs = F.softmax(logits, dim=1)

    # Calculate pixel-wise entropy: H(p) = -sum(p * log(p))
    # A small epsilon is added to prevent log(0).
    uncertainty_map = -torch.sum(softmax_probs * torch.log(softmax_probs.clamp_min(1e-8)), dim=1)

    # Normalize entropy to a [0, 1] range for stable weighting
    uncertainty_map = uncertainty_map / np.log(logits.shape[1])  # Divide by max possible entropy

    # --- 3. Combine the loss with the uncertainty weight ---
    # We create a dynamic weight: (1 + uncertainty). This means that for pixels with high
    # uncertainty, the loss is amplified. For pixels with low uncertainty, the loss is close to the original.
    dynamic_weights = (1 + uncertainty_map).unsqueeze(1)  # Add channel dimension for broadcasting

    # Apply the dynamic weights to the pixel-wise BCE loss
    weighted_bce_loss = bce_loss_pixelwise * dynamic_weights

    # Calculate the final mean loss, considering only valid (non-ignored) pixels
    total_loss = (weighted_bce_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    # --- 4. Metric Calculation (only for validation) ---
    if not calculate_metrics:
        return total_loss, 0.0, 0.0

    with torch.no_grad():
        # Use sigmoid probabilities for the BCE-based metrics
        sigmoid_probs = torch.sigmoid(logits)

        confidence_thresholds = np.linspace(0.1, 0.9, 20)
        distance_threshold = 30

        probs_np = sigmoid_probs.cpu().numpy()
        labels_np = one_hot_labels.cpu().numpy()

        probs_mitotic = probs_np[:, 1, :, :]
        probs_non_mitotic = probs_np[:, 2, :, :]

        gt_mitotic = labels_np[:, 1, :, :]
        gt_non_mitotic = labels_np[:, 2, :, :]

        f1_mitotic = compute_best_f1(probs_mitotic, gt_mitotic, confidence_thresholds, distance_threshold)
        f1_non_mitotic = compute_best_f1(probs_non_mitotic, gt_non_mitotic, confidence_thresholds, distance_threshold)

    return total_loss, f1_mitotic, f1_non_mitotic

