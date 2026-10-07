import numpy as np
from pathlib import Path
import cv2, json
import SimpleITK as sitk
from PIL import Image

MEAN = np.array([0.709925, 0.444565, 0.723188], np.float32)
STD  = np.array([0.182734, 0.221381, 0.144816], np.float32)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_sitk_image(path: Path):
    return sitk.ReadImage(str(path))

def get_mm_spacing(sitk_img):
    sp = sitk_img.GetSpacing()
    if len(sp) >= 2 and sp[0] > 0 and sp[1] > 0:
        return float(sp[0]), float(sp[1])  # mm per pixel
    return None, None

def to_tensor(img):
    # Expect HWC; coerce to 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 4:    img = img[:, :, :3]
        elif c == 1:  img = np.repeat(img, 3, axis=2)
        elif c > 3:   img = img[:, :, :3]

    x = img.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2,0,1))[None, ...]  # (1,3,H,W)
    import torch
    return torch.from_numpy(x)

def mm_per_px_from_tiff_dpi(path: Path):
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi", None)
            if dpi is None:
                return (0.0001, 0.0001)
            if isinstance(dpi, tuple): xdpi, ydpi = dpi
            else: xdpi = ydpi = float(dpi)
            if xdpi > 0 and ydpi > 0:
                return (25.4 / float(xdpi), 25.4 / float(ydpi))
    except Exception:
        pass
    return (0.0001, 0.0001)

def write_gc_points(out_path: Path, pts_xy_px, p_mitosis, mmx, mmy, thresh):
    points = []
    for (x,y), p in zip(pts_xy_px, p_mitosis):
        points.append({
            "name": "mitotic figure" if p >= thresh else "non-mitotic figure",
            "point": [float(x * mmx), float(y * mmy), 0],
            "probability": float(p)
        })
    payload = {"type": "Multiple points", "points": points, "version": {"major": 1, "minor": 0}}
    out_path.write_text(json.dumps(payload))

