"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""


import numpy as np
from torchvision.transforms import Lambda
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)  # shape [num_classes]
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        log_probs = F.log_softmax(inputs, dim=1)
        log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p = torch.exp(log_p)
        loss = -((1 - p) ** self.gamma) * log_p

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



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
    best_precision = 0
    best_recall = 0

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
            best_precision = precision
            best_recall = recall

    return max_f1, best_precision, best_recall


def calculate_dice_scores(probs, gt_mask, num_classes=3):
    """NEW: Calculates pixel-wise Dice score for each class."""
    dice_scores = []
    # Get final class predictions
    preds = np.argmax(probs, axis=1)  # (N, H, W)

    for i in range(num_classes):
        pred_class = (preds == i)
        true_class = (gt_mask == i)
        intersection = np.sum(pred_class * true_class)
        union = np.sum(pred_class) + np.sum(true_class)
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice)

    return dice_scores


# --- THE CORRECTED MAIN FUNCTION ---
def calculate_loss_and_metrics(logits, mask, focal_loss_fn, calculate_metrics=False):
    """
    Calculates Focal Loss for training and optionally computes a full suite of detection and segmentation metrics.

    Args:
        logits (torch.Tensor): Raw output from the model (N, C, H, W).
        mask (torch.Tensor): Ground truth mask with class indices (N, H, W).
        focal_loss_fn (nn.Module): The instantiated Focal Loss module.
        calculate_metrics (bool): If True, compute the slow detection and segmentation scores.

    Returns:
        A tuple containing all requested scores.
    """
    # 1. Calculate the loss. This is always done and is fast.
    loss = focal_loss_fn(logits, mask.long())

    # 2. If we are not in a validation/test step, return early with zero values for all metrics.
    if not calculate_metrics:
        return loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 3. If metrics are requested, run the slow, CPU-based post-processing.
    with torch.no_grad():
        confidence_thresholds = np.linspace(0.1, 0.9, 20)
        distance_threshold = 30

        # Get probabilities and move to CPU
        probs = F.softmax(logits, dim=1).cpu().numpy()
        gt_mask = mask.cpu().numpy()

        # --- Calculate Detection Metrics (F1, Precision, Recall) ---
        probs_mitotic = probs[:, 1, :, :]
        probs_non_mitotic = probs[:, 2, :, :]

        gt_mitotic = (gt_mask == 1).astype(np.uint8)
        gt_non_mitotic = (gt_mask == 2).astype(np.uint8)

        f1_mitotic, precision_mitotic, recall_mitotic = compute_best_f1(
            probs_mitotic, gt_mitotic, confidence_thresholds, distance_threshold
        )
        f1_non_mitotic, precision_non_mitotic, recall_non_mitotic = compute_best_f1(
            probs_non_mitotic, gt_non_mitotic, confidence_thresholds, distance_threshold
        )

        # --- Calculate Segmentation Metrics (Dice Score) ---
        background_dice, mitotic_dice, non_mitotic_dice = calculate_dice_scores(
            probs, gt_mask
        )

    return (
        loss,
        f1_mitotic,
        f1_non_mitotic,
        precision_mitotic,
        precision_non_mitotic,
        recall_mitotic,
        recall_non_mitotic,
        background_dice,
        mitotic_dice,
        non_mitotic_dice
    )

'''
def Class_Wise_TIL_Detection_FROC(input, labels):
    eps = 1e-8
    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 30
    initial_threshold = 0.1

    predicted_detections1 = []
    ground_truth_detections1 = []
    predicted_detections2 = []
    ground_truth_detections2 = []

    preds = F.softmax(input, dim=1)

    labels_0 = labels[:, 0, :, :]
    labels_1 = labels[:, 1, :, :]
    labels_2 = labels[:, 2, :, :]
    labels_11 = labels_1.detach().float().cpu().numpy()
    labels_22 = labels_2.detach().float().cpu().numpy()

    pred_0 = preds[:, 0, :, :]
    pred_1 = preds[:, 1, :, :]
    pred_2 = preds[:, 2, :, :]
    preds_11 = pred_1.detach().float().cpu().numpy().astype(np.float32, copy=False)
    preds_22 = pred_2.detach().float().cpu().numpy().astype(np.float32, copy=False)

    labels_0 = labels_0.contiguous().view(-1)
    labels_1 = labels_1.contiguous().view(-1)
    labels_2 = labels_2.contiguous().view(-1)

    pred_0 = pred_0.contiguous().view(-1)
    pred_1 = pred_1.contiguous().view(-1)
    pred_2 = pred_2.contiguous().view(-1)

    intersection_0 = (pred_0 * labels_0).sum()
    intersection_1 = (pred_1 * labels_1).sum()
    intersection_2 = (pred_2 * labels_2).sum()
    intersection_SUM = intersection_0 + intersection_1 + intersection_2

    union_0 = (pred_0.sum() + labels_0.sum()) + eps
    union_1 = (pred_1.sum() + labels_1.sum()) + eps
    union_2 = (pred_2.sum() + labels_2.sum()) + eps
    union_SUM = union_0 + union_1 + union_2

    dice_total = (2 * intersection_SUM / union_SUM)
    #dice_total_loss = 1 - dice_total

    dice0 = (2 * intersection_0 / union_0)
    dice1 = (2 * intersection_1 / union_1)
    dice2 = (2 * intersection_2 / union_2)
    #diceT = dice0 + dice1 + dice2

    alpha = [0.05, 0.475, 0.475]
    focalloss = FocalLossMultiClass(alpha=alpha, gamma=2.0)

    loss = focalloss(preds, labels)

    #bce_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')

    for i in range(len(preds_11)):
        pred_temp1 = extract_predictions(preds_11[i], confidence_threshold=initial_threshold)
        predicted_detections1.append(non_max_supression_distance(pred_temp1, distance_threshold=50))

    for i in range(len(preds_22)):
        pred_temp2 = extract_predictions(preds_22[i], confidence_threshold=initial_threshold)
        predicted_detections2.append(non_max_supression_distance(pred_temp2, distance_threshold=50))

    for i in range(len(labels_11)):
        label_temp1 = extract_true_centers(labels_11[i], confidence_threshold=initial_threshold, min_distance=10)
        ground_truth_detections1.append(label_temp1)

    for i in range(len(labels_22)):
        label_temp2 = extract_true_centers(labels_22[i], confidence_threshold=initial_threshold, min_distance=10)
        ground_truth_detections2.append(label_temp2)

    f1_1, prec_1, recall_1 = compute_best_f1(predicted_detections1, ground_truth_detections1, confidence_thresholds, distance_threshold)
    f1_2, prec_2, recall_2 = compute_best_f1(predicted_detections2, ground_truth_detections2, confidence_thresholds, distance_threshold)


    return loss, f1_1, f1_2, prec_1, prec_2, recall_1, recall_2, dice0, dice1, dice2

'''