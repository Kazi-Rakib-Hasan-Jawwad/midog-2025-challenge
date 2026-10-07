import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def nms_points(pts_xy: np.ndarray, scores: np.ndarray, radius_px: int):
    """
    Greedy Euclidean NMS for point detections.
    Keeps highest-score point and suppresses others within radius_px.
    pts_xy: (N,2) in pixels, scores: (N,)
    """
    if pts_xy.size == 0:
        return pts_xy, scores
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for idx_k, i in enumerate(order):
        if suppressed[idx_k]:
            continue
        keep.append(i)
        # suppress points too close to i
        dx = pts_xy[order, 0] - pts_xy[i, 0]
        dy = pts_xy[order, 1] - pts_xy[i, 1]
        suppressed |= (dx*dx + dy*dy) <= (radius_px * radius_px)
        suppressed[idx_k] = False  # keep the winner itself
    keep = np.array(keep, dtype=int)
    return pts_xy[keep], scores[keep]

def find_points_from_two_heads(logits, min_prob=0.05, min_dist=20, mitotic_idx=1, non_mitotic_idx=2):
    """
    logits: (C,H,W) pre-sigmoid, ch1=mitotic, ch2=non-mitotic
    candidate_mode:
      - 'union':  U = max(sigmoid(L1), sigmoid(L2))       (default; robust)
    Returns:
      pts_xy (N,2), p_mitosis (N,)
    """
    L1 = logits[mitotic_idx]
    L2 = logits[non_mitotic_idx]
    P1 = 1.0 / (1.0 + np.exp(-L1))
    P2 = 1.0 / (1.0 + np.exp(-L2))

    U = np.maximum(P1, P2)

    mask = U >= min_prob
    if mask.sum() == 0:
        return np.zeros((0,2), np.float32), np.zeros((0,), np.float32)

    # distance transform for separation
    dist = ndi.distance_transform_edt(mask)
    local_max = peak_local_max(dist, indices=False, min_distance=max(1,int(0.9*min_dist)), labels=mask)
    if local_max.sum() == 0:
        return np.zeros((0,2), np.float32), np.zeros((0,), np.float32)

    markers, _ = ndi.label(local_max)
    labels = watershed(-dist, markers, mask=mask)

    pts, p_mitosis = [], []
    for lab in range(1, labels.max()+1):
        ys, xs = np.where(labels == lab)
        if len(xs) == 0:
            continue
        # choose pixel with highest union prob for stability
        idx = np.argmax(U[ys, xs])
        x, y = xs[idx], ys[idx]

        # binary probability via logit difference
        p = 1.0 / (1.0 + np.exp(-(L1[y,x] - L2[y,x])))
        pts.append([float(x), float(y)])
        p_mitosis.append(float(p))
    return np.array(pts, np.float32), np.array(p_mitosis, np.float32)

