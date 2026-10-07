import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def find_points_from_two_heads(logits, min_prob=0.05, min_dist=20, candidate_mode="union", mitotic_idx=1, non_mitotic_idx=2):
    """
    logits: (C,H,W) pre-sigmoid, ch1=mitotic, ch2=non-mitotic
    candidate_mode:
      - 'union':  U = max(sigmoid(L1), sigmoid(L2))       (default; robust)
      - 'diff' :  U = sigmoid(L1 - L2)                    (discriminative)
      - 'both' :  U = max( sigmoid(L1 - L2), max(P1,P2) ) (catch weak-but-separable)
    Returns:
      pts_xy (N,2), p_mitosis (N,)
    """
    L1 = logits[1];
    L2 = logits[2]
    P1 = 1.0 / (1.0 + np.exp(-L1))
    P2 = 1.0 / (1.0 + np.exp(-L2))
    diff = 1.0 / (1.0 + np.exp(-(L1 - L2)))  # sigmoid(L1-L2)

    if candidate_mode == "union":
        U = np.maximum(P1, P2)
    elif candidate_mode == "diff":
        U = diff
    else:  # 'both'
        U = np.maximum(diff, np.maximum(P1, P2))

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

