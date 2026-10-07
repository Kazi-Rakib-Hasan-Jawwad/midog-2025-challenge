# utils/seg_to_det.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy import ndimage as ndi
from inspect import signature

# skimage API changed around 0.20; keep compatibility with 0.17.2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Try both constructor signatures you might have
def _build_deeplab(num_classes: int):
    from utils.DeepLabv3_plus import DeepLabv3Plus
    try:
        return DeepLabv3Plus(num_classes=num_classes)
    except TypeError:
        return DeepLabv3Plus(num_classes_det=num_classes)

def _clean_state_dict(sd: dict) -> dict:
    """
    Strip common prefixes from PL/DP/DDP saves: 'net.', 'model.', 'module.', 'seg.'.
    """
    prefixes = ("net.", "model.", "module.", "seg.", "DeepLabv3Plus.")
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _peak_coords(distance: np.ndarray, binary: np.ndarray, min_dist_px: int) -> np.ndarray:
    params = signature(peak_local_max).parameters
    if "indices" in params:  # old API (e.g., 0.17.2)
        coords = peak_local_max(distance, min_distance=min_dist_px, indices=True, labels=binary)
    else:  # new API (>=0.20)
        coords = peak_local_max(distance, min_distance=min_dist_px, labels=binary)
    return coords  # rows, cols (y, x)

def _extract_centers(prob_map: np.ndarray, min_prob: float, min_dist_px: int) -> np.ndarray:
    binary = prob_map >= min_prob
    if not binary.any():
        return np.zeros((0, 2), dtype=np.int32)

    dist = ndi.distance_transform_edt(binary)
    coords = _peak_coords(dist, binary, min_dist_px)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    labels = watershed(-dist, markers, mask=binary)

    centers = []
    for i in range(1, labels.max() + 1):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue
        idx = np.argmax(dist[ys, xs])
        centers.append([int(xs[idx]), int(ys[idx])])
    return np.array(centers, dtype=np.int32)

class SegToDetWrapper(nn.Module):
    """
    Wrap your semantic segmentation model so it mimics a Torchvision detector:
    returns list[{'boxes','scores','labels'}] per image.
    """
    def __init__(
        self,
        checkpoint: str | Path,
        mitotic_channel: int = 1,
        min_prob_init: float = 0.09,
        min_distance_px_seed: int = 3, # seed separation in px; tiler/NMS will refine; px→mm is done later
        seed_box_radius_px=12
    ):
        super().__init__()
        self.seg = _build_deeplab(num_classes=3)
        self.mitotic_channel = int(mitotic_channel)
        self.min_prob_init = float(min_prob_init)
        self.min_distance_px_seed = int(min_distance_px_seed)
        self.seed_box_radius_px = int(seed_box_radius_px)
        self._channel_locked = False

        ckpt = torch.load(str(checkpoint), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        keys = list(state.keys())[:5]
        print(f"[ckpt] top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        print(f"[ckpt] state_dict sample: {keys}")

        res = self.seg.load_state_dict(_clean_state_dict(state), strict=False)
        try:
            missing = getattr(res, 'missing_keys', [])
            unexpected = getattr(res, 'unexpected_keys', [])
        except Exception:
            missing = unexpected = []
        print(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing and len(missing) > 50:
            print("[ckpt][WARN] Many missing keys — architecture/name mismatch likely. Check prefixes and model class.")
        #self.seg.load_state_dict(_clean_state_dict(state), strict=False)
        self.seg.eval()

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        x = torch.stack(images, dim=0)      # [B,3,H,W]
        logits = self.seg(x)                # [B,3,H,W]
        probs  = torch.sigmoid(logits)
        # Optional channel auto‑selection (run per batch, cheap)
        if getattr(self, 'auto_select_channel', False)  and not self._channel_locked:
            ch_scores = []
            for ch_i in range(probs.shape[1]):
                p = probs[0, ch_i].detach().cpu().numpy().astype('float32', copy=False)
                # proxy: number of local maxima above a tiny floor + their mean value
                from numpy import mean
                centers = _extract_centers(p, max(1e-3, self.min_prob_init), self.min_distance_px_seed)
                score = len(centers) + float(p[p > 0.1].mean() if (p > 0.1).any() else 0.0)
                ch_scores.append(score)
            self.mitotic_channel = int(np.argmax(ch_scores))
            self._channel_locked = True
            print(f"[seg2det] auto-selected mitotic_channel={self.mitotic_channel}")
        B, _, H, W = probs.shape
        ch = self.mitotic_channel
        outs: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            p = probs[b, ch].detach().cpu().numpy().astype(np.float32, copy=False)
            centers_xy = _extract_centers(p, self.min_prob_init, self.min_distance_px_seed)

            if centers_xy.shape[0] == 0:
                outs.append({
                    "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                })
                continue

            # tiny box around each center so the stock tiler/NMS can run
            r = self.seed_box_radius_px
            x1 = np.clip(centers_xy[:, 0] - r, 0, W-1)
            y1 = np.clip(centers_xy[:, 1] - r, 0, H-1)
            x2 = np.clip(centers_xy[:, 0] + r, 0, W-1)
            y2 = np.clip(centers_xy[:, 1] + r, 0, H-1)
            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            scores = p[centers_xy[:, 1], centers_xy[:, 0]].astype(np.float32)

            outs.append({
                "boxes":  torch.from_numpy(boxes),
                "scores": torch.from_numpy(scores),
                "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            })
        return outs