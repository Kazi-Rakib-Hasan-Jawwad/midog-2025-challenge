#!/usr/bin/env python3
"""
Style discovery + representative selection + (optional) YAML stats for RandStainNA.

What this script does
---------------------
1) Scans a source folder of patch images.
2) Extracts style features (LAB/HSV/HED channel stats + FFT power ratios).
3) Chooses the number of clusters automatically using silhouette score (k in [--min_k, --max_k]).
4) Creates destination folders style1, style2, ... and saves N representative images per style (closest to cluster centroids).
5) (Optional) Generates a RandStainNA-compatible YAML by running your provided
   `preprocessing_statistics.py` against the created `train/` directory.

Example
-------
python style_select_and_yaml.py \
  --src "/home/rakib/data/MIDOGpp-main/224/sorted_data/folder5/img" \
  --dst "/home/rakib/PycharmProjects/MIDOG2025_MICCAI/stain_augmentation/train" \
  --min_k 4 --max_k 12 --per_style 2 \
  --run_yaml \
  --preproc "/mnt/data/preprocessing_statistics.py" \
  --yaml_out "/home/rakib/PycharmProjects/MIDOG2025_MICCAI/stain_augmentation/output" \
  --dataset_name folder5_styles

Dependencies
------------
- numpy, opencv-python (or opencv-python-headless), scikit-image, scikit-learn, pyyaml

Notes
-----
- Images are treated as RGB after cv2.imread(... )[:, :, ::-1].
- We downscale to 224 on feature extraction for speed if images are larger.
"""
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import color
import warnings


def list_images(src_dir):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = [str(p) for p in Path(src_dir).glob("**/*") if p.suffix.lower() in exts]
    return sorted(paths)


def safe_read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize_max_side(img, max_side=224):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def fft_power_ratio(gray):
    """Return low/mid/high frequency power ratios as 3 features.
    Splits the shifted FFT magnitude by radial thresholds.
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - h/2.0)**2 + (xx - w/2.0)**2)
    rmax = rr.max() + 1e-6
    # thresholds at 0.33 and 0.66 of max radius
    t1, t2 = 0.33*rmax, 0.66*rmax
    low = mag[rr <= t1].sum()
    mid = mag[(rr > t1) & (rr <= t2)].sum()
    high = mag[rr > t2].sum()
    s = low + mid + high + 1e-8
    return np.array([low/s, mid/s, high/s], dtype=np.float32)


def channel_stats(img):
    """Return mean/std for LAB, HSV, and HED (Hematoxylin-Eosin-DAB) channels.
    Produces 18 features: (LAB 6) + (HSV 6) + (HED 6).
    """
    img_f = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor((img_f*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor((img_f*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hed = color.rgb2hed(img_f).astype(np.float32)
    feats = []
    for arr in (lab, hsv, hed):
        if arr.ndim == 2:
            arr = arr[..., None]
        for c in range(arr.shape[2]):
            ch = arr[..., c]
            feats.append(float(ch.mean()))
            feats.append(float(ch.std()))
    return np.array(feats, dtype=np.float32)


def extract_features(paths, max_side=224):
    X = []
    for p in tqdm(paths, desc="Extracting features"):
        img = safe_read_rgb(p)
        img = resize_max_side(img, max_side=max_side)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_fft = fft_power_ratio(gray)
        f_stats = channel_stats(img)
        feat = np.concatenate([f_stats, f_fft], axis=0)  # 18 + 3 = 21 features
        X.append(feat)
    X = np.vstack(X)
    # standardize per-feature
    mu, sigma = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    return Xn


def pick_k(X, kmin=4, kmax=12, seed=42):
    best_k, best_score, best_model = None, -1.0, None
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_model = k, score, km
    if best_model is None:
        # fallback to k=4
        best_model = KMeans(n_clusters=max(2, kmin), n_init=10, random_state=seed).fit(X)
        best_k = best_model.n_clusters
        best_score = float('nan')
    return best_k, best_score, best_model


def closest_indices_to_centroids(X, km, n_per_cluster=2):
    centers = km.cluster_centers_
    labels = km.labels_
    idxs = []
    for c in range(km.n_clusters):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) == 0:
            continue
        # distances to centroid
        d = np.linalg.norm(X[cluster_idx] - centers[c][None, :], axis=1)
        order = np.argsort(d)
        take = min(n_per_cluster, len(cluster_idx))
        idxs.extend(cluster_idx[order[:take]].tolist())
    return idxs


def save_representatives(paths, labels, chosen_idx, dst_root, per_style=2):
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    # map from cluster -> list of chosen global idx within that cluster order
    cluster_to_members = {}
    for gi in chosen_idx:
        c = int(labels[gi])
        cluster_to_members.setdefault(c, []).append(gi)
    # ensure only top-N per cluster (in case of duplicates due to edge cases)
    for c in cluster_to_members:
        cluster_to_members[c] = cluster_to_members[c][:per_style]
    # write files
    created = []
    for c, gidxs in sorted(cluster_to_members.items()):
        style_dir = dst_root / f"style{c+1}"
        style_dir.mkdir(parents=True, exist_ok=True)
        for j, gi in enumerate(gidxs, start=1):
            src = Path(paths[gi])
            dst = style_dir / f"rep{j}_{src.name}"
            shutil.copy2(src, dst)
            created.append(str(dst))
    return created


def generate_yaml_with_preproc(train_dir, preproc_path, yaml_out, dataset_name="dataset", color_space="LAB", methods="Reinhard", randomize=True, n_each_class=0):
    """Run the user's preprocessing_statistics.py with custom parameters via runpy."""
    import runpy
    # Prepare globals that the script expects
    g = {
        'path_dataset': str(train_dir) if str(train_dir).endswith('/') else str(train_dir) + '/',
        'save_dir': str(yaml_out),
        'dataset_name': str(dataset_name),
        'methods': str(methods),
        'color_space': str(color_space),
        'randomize': bool(randomize),
        'n': int(n_each_class),
    }
    print(f"[YAML] Running preprocessing on: {g['path_dataset']} -> {yaml_out} ({dataset_name})")
    runpy.run_path(str(preproc_path), init_globals=g)


def main():
    ap = argparse.ArgumentParser(description="Discover style clusters and generate RandStainNA YAML.")
    ap.add_argument('--src', default="/home/rakib/data/MIDOGpp-main/224/sorted_data/folder5/img") #, required=True, help='Source folder with images',
    ap.add_argument('--dst', default="/home/rakib/PycharmProjects/MIDOG2025_MICCAI/stain_augmentation/train/") # required=True, help='Destination train/ folder where styleX subfolders will be created'
    ap.add_argument('--min_k', type=int, default=4)
    ap.add_argument('--max_k', type=int, default=8)
    ap.add_argument('--per_style', type=int, default=2, help='Representative images per style folder')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_side', type=int, default=224, help='Max side used for feature extraction')
    ap.add_argument('--run_yaml', action='store_true', help='Run preprocessing_statistics.py afterwards')
    ap.add_argument('--preproc', type=str, default='', help='Path to preprocessing_statistics.py')
    ap.add_argument('--yaml_out', type=str, default='', help='Output folder for YAML file')
    ap.add_argument('--dataset_name', type=str, default='random_images')
    ap.add_argument('--color_space', type=str, default='LAB', choices=['LAB','HED','HSV'])
    ap.add_argument('--methods', type=str, default='Reinhard')
    ap.add_argument('--n_each_class', type=int, default=0, help='0 means use all images in each style class')

    args = ap.parse_args()

    paths = list_images(args.src)
    if not paths:
        raise SystemExit(f"No images found under: {args.src}")
    print(f"Found {len(paths)} images in {args.src}")

    X = extract_features(paths, max_side=args.max_side)
    k, score, km = pick_k(X, kmin=args.min_k, kmax=args.max_k, seed=args.seed)
    print(f"Chosen k={k} (silhouette={score:.4f})")

    labels = km.labels_
    chosen_idx = closest_indices_to_centroids(X, km, n_per_cluster=args.per_style)

    created = save_representatives(paths, labels, chosen_idx, args.dst, per_style=args.per_style)
    print(f"Saved {len(created)} representative images to {args.dst}/style*/")

    # Optionally generate YAML using user's script
    if args.run_yaml:
        if not args.preproc or not os.path.isfile(args.preproc):
            raise SystemExit("--run_yaml specified but --preproc path is invalid.")
        yaml_out = args.yaml_out if args.yaml_out else str(Path(args.dst).parent / 'output')
        os.makedirs(yaml_out, exist_ok=True)
        generate_yaml_with_preproc(
            train_dir=args.dst,
            preproc_path=args.preproc,
            yaml_out=yaml_out,
            dataset_name=args.dataset_name,
            color_space=args.color_space,
            methods=args.methods,
            randomize=True,
            n_each_class=args.n_each_class,
        )
        print("YAML generation finished.")


if __name__ == '__main__':
    main()
