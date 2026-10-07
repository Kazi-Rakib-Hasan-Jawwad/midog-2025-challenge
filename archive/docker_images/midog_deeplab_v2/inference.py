# imports (add SITK helpers)
import os, argparse, json, yaml
from pathlib import Path
import torch, numpy as np
import SimpleITK as sitk  # <-- add
from algorithm.loader import load_deeplab_from_ckpt
from algorithm.utils import (
    ensure_dir, to_tensor, write_gc_points,
    read_sitk_image, get_mm_spacing, mm_per_px_from_tiff_dpi
)
from algorithm.tiling import slide_logits
from algorithm.postproc import find_points_from_two_heads
from glob import glob

ALLOWED_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".mha",".svs",".ndpi"}

def _allowed(p):
    return p.suffix.lower() in ALLOWED_EXTS

def load_yaml(path: Path, defaults: dict) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict): return defaults
        out = defaults.copy(); out.update(data); return out
    except Exception:
        return defaults

def find_image_via_gc_interface(in_dir: Path) -> Path:
    """Grand-Challenge style: /input/inputs.json lists socket slugs; image under /input/images/<slug>/..."""
    jj = in_dir / "inputs.json"
    if not jj.exists(): return None
    try:
        inputs = json.loads(jj.read_text())
    except Exception:
        return None
    # collect slugs
    slugs = []
    for s in inputs:
        iface = s.get("interface") or {}
        slug = iface.get("slug")
        if slug: slugs.append(slug)
    # try exact slugs
    for slug in slugs:
        d = in_dir / "images" / slug
        if d.exists():
            files = [p for p in sorted(d.iterdir()) if p.is_file() and _allowed(p)]
            if files: return files[0]
    # compatibility: scan one level down
    img_root = in_dir / "images"
    if img_root.exists():
        for sub in sorted(img_root.iterdir()):
            if sub.is_dir():
                files = [p for p in sorted(sub.iterdir()) if p.is_file() and _allowed(p)]
                if files: return files[0]
    return None

def find_image_fallback(in_dir: Path) -> Path:
    """Local/dev style: look directly under /input or /input/images"""
    img_dir = in_dir / "images" if (in_dir / "images").exists() else in_dir
    files = [p for p in sorted(img_dir.iterdir()) if p.is_file() and _allowed(p)]
    return files[0] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=None)
    # post-proc & tiling knobs
    ap.add_argument("--tile-size", type=int, default=int(os.getenv("TILE_SIZE", "1024")))
    ap.add_argument("--stride", type=int, default=int(os.getenv("STRIDE", "614")))
    ap.add_argument("--peak-min", type=float, default=float(os.getenv("PEAK_MIN", "0.05")))
    ap.add_argument("--peak-dist", type=int, default=int(os.getenv("PEAK_DIST", "18")))
    ap.add_argument("--mitotic-thresh", type=float, default=float(os.getenv("MITOTIC_THRESH", "0.52")))
    ap.add_argument("--candidate-mode", choices=["union", "diff", "both"],
                    default=os.getenv("CANDIDATE_MODE", "union"))
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    # --- Load organizer-style configs from resources/ ---
    res = Path("/opt/algorithm/resources")  # baked into image by Dockerfile
    patch_cfg = load_yaml(res / "patch_config.yaml", {"tile_size": 1024, "overlap": 0.4})
    model_cfg = load_yaml(res / "model_config.yaml", {"checkpoint": "last.ckpt", "mitotic_channel": 1, "non_mitotic_channel": 2})
    infer_cfg = load_yaml(res / "inference_config.yaml", {"candidate_mode": "union", "peak_min": 0.05, "peak_dist": 18, "mitotic_thresh": 0.52})

    # compute stride from overlap unless overridden
    tile_size = args.tile_size or int(os.getenv("TILE_SIZE", patch_cfg["tile_size"]))
    if "stride" in patch_cfg and patch_cfg["stride"] is not None:
        stride_default = int(patch_cfg["stride"])
    else:
        stride_default = int(round(tile_size * (1.0 - float(patch_cfg.get("overlap", 0.4)))))
    stride = args.stride or int(os.getenv("STRIDE", stride_default))

    # post-proc knobs
    candidate_mode = args.candidate_mode or os.getenv("CANDIDATE_MODE", infer_cfg["candidate_mode"])
    peak_min = args.peak_min or float(os.getenv("PEAK_MIN", infer_cfg["peak_min"]))
    peak_dist = args.peak_dist or int(os.getenv("PEAK_DIST", infer_cfg["peak_dist"]))
    mit_thr = args.mitotic_thresh or float(os.getenv("MITOTIC_THRESH", infer_cfg["mitotic_thresh"]))

    # --- Resolve weights from organizer location first ---
    ckpt_name = model_cfg.get("checkpoint", "last.ckpt")
    if args.weights:
        weights = Path(args.weights)
    else:
        weights = Path("/opt/ml/model") / ckpt_name
        if not weights.exists():
            weights = res / ckpt_name
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = load_deeplab_from_ckpt(weights)
    device = next(model.parameters()).device

    # --- Find the input image per GC interface ---
    img_path = find_image_via_gc_interface(in_dir)
    if img_path is None:
        img_path = find_image_fallback(in_dir)
    if img_path is None:
        raise FileNotFoundError("No input image found via GC interface or local fallback.")

    # Load & normalize via SimpleITK
    img_sitk = read_sitk_image(img_path)
    mmx, mmy = get_mm_spacing(img_sitk)  # SITK spacing
    np_img = sitk.GetArrayFromImage(img_sitk)   # (H,W) or (C,H,W)

    if np_img.ndim == 2:
        np_img = np.stack([np_img]*3, axis=-1)
    elif np_img.ndim == 3:
        if np_img.shape[0] in (1, 3, 4) and np_img.shape[2] not in (1, 3, 4):
            np_img = np.transpose(np_img, (1, 2, 0))
        # Coerce to 3-channel RGB
        if np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]
        elif np_img.shape[2] == 1:
            np_img = np.repeat(np_img, 3, axis=2)

    x = to_tensor(np_img).to(device)

    # Sliding-window logits with blending
    logits = slide_logits(model, x, tile=tile_size, stride=stride)[0]  # (C,H,W) numpy

    # Candidate extraction
    pts_xy, p_mitosis = find_points_from_two_heads(
        logits, min_prob=peak_min, min_dist=peak_dist, candidate_mode=args.candidate_mode, mitotic_idx=int(model_cfg.get("mitotic_channel",1)),
        non_mitotic_idx=int(model_cfg.get("non_mitotic_channel",2))
    )

    if not (mmx and mmy) or mmx <= 0 or mmy <= 0:
        mmx, mmy = mm_per_px_from_tiff_dpi(img_path)
    # define output path and write JSON
    out_path = out_dir / "mitotic-figures.json"
    write_gc_points(out_path, pts_xy, p_mitosis, mmx, mmy, mit_thr)

if __name__ == "__main__":
    main()

