import math, numpy as np, torch
from typing import Tuple

@torch.no_grad()
def slide_logits(model, img_tensor, tile=1024, stride=614):
    """
    img_tensor: (1,3,H,W) float32 on device
    returns blended logits (1,C,H,W) on CPU numpy
    """
    _, _, H, W = img_tensor.shape
    device = next(model.parameters()).device
    C = 3  # your model has 3 channels

    out = torch.zeros((1, C, H, W), device=device)
    acc = torch.zeros((1, 1, H, W), device=device)
    
    tile_h = min(tile, H)
    tile_w = min(tile, W)
    
    # compute start positions that guarantee full coverage (no negatives)
    ys = list(range(0, max(1, H - tile_h + 1), stride))
    xs = list(range(0, max(1, W - tile_w + 1), stride))
    if ys[-1] != H - tile_h:
        ys.append(H - tile_h)
    if xs[-1] != W - tile_w:
        xs.append(W - tile_w)
    
    for y0 in ys:
        y1 = y0 + tile_h
        for x0 in xs:
            x1 = x0 + tile_w
            patch = img_tensor[:, :, y0:y1, x0:x1]
            logits = model(patch)  # (1,C,h,w), h<=tile_h, w<=tile_w
            out[:, :, y0:y1, x0:x1] += logits
            acc[:, :, y0:y1, x0:x1] += 1.0

    out = out / acc.clamp_min(1.0)
    return out.detach().cpu().numpy()

