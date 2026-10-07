#!/usr/bin/env python3
"""Reconstruct MIDOG++ segmentation patches from the recovered code and paper.

This is a documented reconstruction, not a claim of byte-identical recovery.
Images and annotations are read only. All generated artifacts stay in --output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import PIL
from PIL import Image

PATCH = 512
VISIBILITY = 0.8
IMAGE_FRACTION = 0.8
RECIPE = {
    'schema_version': 1,
    'patch_size': PATCH,
    'bbox_format': 'xyxy (verified 50x50 boxes, NOT COCO xywh)',
    'grid': 'n=round(1+dimension/512); start=round(i*(dimension-512)/(n-1)) using exact rational arithmetic',
    'minimum_object_visibility': VISIBILITY,
    'minimum_valid_image_fraction': IMAGE_FRACTION,
    'filter': 'discard empty patches and any patch containing an object below minimum visibility',
    'raster_boundary_rule': 'include every annotation whose original inclusive OpenCV raster touches the patch; zero-area geometric touches fail visibility',
    'mask': 'original JSON order, int32 coordinate truncation, cv2.fillPoly inclusive endpoints; later annotations overwrite earlier ones',
    'mask_labels': {'0': 'background', '1': 'mitotic figure', '2': 'not mitotic figure'},
    'image': 'original-resolution RGB, no resizing, no stain or intensity transformation; alpha>0 denotes source coverage',
    'png_compress_level': 1,
    'split': 'upstream datasets_xvalidation.csv train/test image assignments; no new validation split',
    'historical_identity': 'reconstructed from archived mask/grid code and contribution paper; original 512_seg_root generator unavailable',
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def axis_starts(length, patch=PATCH):
    if length <= patch:
        return [0]
    count = max(2, round(1 + length / patch))
    return [round(Fraction(i * (length - patch), count - 1)) for i in range(count)]


def grid(width, height):
    index = 0
    for top in axis_starts(height):
        for left in axis_starts(width):
            index += 1
            yield index, (left, top, left + PATCH, top + PATCH)


def selection(annotations, box):
    """Visibility uses original continuous box area, before image clipping."""
    left, top, right, bottom = box
    selected, fractions = [], []
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        rx1, ry1, rx2, ry2 = (int(v) for v in ann['bbox'])
        if rx2 < left or rx1 >= right or ry2 < top or ry1 >= bottom:
            continue
        area = (x2 - x1) * (y2 - y1)
        visible = max(0, min(x2, right) - max(x1, left)) * max(0, min(y2, bottom) - max(y1, top)) / area
        selected.append(ann)
        fractions.append(visible)
    if not selected:
        return 'empty', [], []
    if min(fractions) + 1e-12 < VISIBILITY:
        return 'object_visibility', selected, fractions
    return 'keep', selected, fractions


def render_mask(width, height, annotations):
    mask = np.zeros((height, width), np.uint8)
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        cv2.fillPoly(mask, [polygon], color=int(ann['category_id']))
    return mask


def load_sources(source, split_csv):
    source = Path(source).resolve()
    ann_path = source / 'databases/MIDOG++.json'
    ann_bytes = ann_path.read_bytes()
    data = json.loads(ann_bytes)
    by_image = defaultdict(list)
    ids = set()
    images = {im['id']: im for im in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('Duplicate image IDs')
    for ann in data['annotations']:
        if ann['id'] in ids or ann['image_id'] not in images:
            raise ValueError('Invalid/duplicate annotation identity')
        ids.add(ann['id'])
        b = ann['bbox']
        if len(b) != 4 or not all(math.isfinite(v) for v in b) or b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError(f'Invalid xyxy box: {ann["id"]}')
        if not math.isclose(b[2] - b[0], 50) or not math.isclose(b[3] - b[1], 50):
            raise ValueError('Annotation format differs from the verified MIDOG++ 50x50 xyxy release')
        if ann['category_id'] not in (1, 2):
            raise ValueError('Unexpected annotation category')
        by_image[ann['image_id']].append(ann)
    with Path(split_csv).open() as f:
        split_rows = list(csv.DictReader(f, delimiter=';'))
    splits = {int(row['Slide']): row['Dataset'] for row in split_rows}
    if len(splits) != len(split_rows) or not set(splits.values()) <= {'train', 'test'}:
        raise ValueError('Unexpected upstream split file')
    present, excluded = [], []
    for im in sorted(images.values(), key=lambda x: x['id']):
        if Path(im['file_name']).name != im['file_name']:
            raise ValueError('Unsafe image filename')
        path = source / 'images' / im['file_name']
        if not path.is_file():
            if by_image[im['id']]:
                raise FileNotFoundError(f'Missing annotated image: {path}')
            excluded.append(im['id'])
            continue
        if im['id'] not in splits:
            raise ValueError(f'No upstream split for {path.name}')
        with Image.open(path) as image:
            if image.size != (im['width'], im['height']):
                raise ValueError(f'Image/annotation dimensions disagree: {path.name}')
            if image.mode not in ('RGB', 'RGBA'):
                raise ValueError(f'Unexpected mode: {path.name}: {image.mode}')
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f'Nontrivial TIFF orientation: {path.name}')
        present.append({**im, 'split': splits[im['id']]})
    binding = {'recipe': RECIPE, 'source': str(source), 'annotation_sha256': digest_bytes(ann_bytes),
               'split_csv': str(Path(split_csv).resolve()), 'split_sha256': digest_file(split_csv),
               'builder_sha256': digest_file(__file__), 'images': present,
               'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': PIL.__version__, 'opencv': cv2.__version__},
               'excluded_missing_unannotated_image_ids': excluded}
    return binding, by_image


def make_plan(binding, by_image):
    totals = Counter()
    represented = set()
    patch_classes = Counter()
    for im in binding['images']:
        for _, box in grid(im['width'], im['height']):
            totals['candidates'] += 1
            reason, objects, _ = selection(by_image[im['id']], box)
            totals[reason] += 1
            if reason == 'keep':
                represented.update(a['id'] for a in objects)
                patch_classes['+'.join(map(str, sorted({a['category_id'] for a in objects})))] += 1
    total_anns = sum(len(v) for v in by_image.values())
    return {'binding': binding, 'geometric_plan': dict(totals), 'patch_annotation_classes': dict(patch_classes),
            'annotations_represented_before_alpha_filter': len(represented),
            'annotations_unrepresented_before_alpha_filter': total_anns - len(represented),
            'uncompressed_output_upper_gib': totals['keep'] * PATCH**2 * 4 / 2**30,
            'note': 'Geometric preview; image-alpha coverage is checked while reading each TIFF.'}


def save_png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', compress_level=1)
    content = buffer.getvalue()
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(content)
    os.replace(temp, path)
    return digest_bytes(content), len(content)


def process_image(source, output, image_info, annotations):
    began = time.monotonic()
    image_id = image_info['id']
    path = source / 'images' / image_info['file_name']
    before = path.stat()
    raw = path.read_bytes()
    raw_sha = digest_bytes(raw)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        alpha = np.asarray(image.getchannel('A')) if image.mode == 'RGBA' else None
        rgb = image.convert('RGB')
    del raw
    full_mask = render_mask(image_info['width'], image_info['height'], annotations)
    counts, classes = Counter(), Counter()
    histogram = np.zeros((3, 256), np.int64)
    patches, candidates = [], []
    for index, box in grid(image_info['width'], image_info['height']):
        left, top, right, bottom = box
        reason, objects, fractions = selection(annotations, box)
        counts['candidates'] += 1
        candidate = {'grid_index': index, 'box': box, 'reason': reason}
        if reason != 'keep':
            counts[reason] += 1
            candidates.append(candidate)
            continue
        valid_w, valid_h = min(right, rgb.width) - left, min(bottom, rgb.height) - top
        coverage = valid_w * valid_h / PATCH**2
        if alpha is not None:
            coverage = np.count_nonzero(alpha[top:bottom, left:right]) / PATCH**2
        candidate['valid_image_fraction'] = coverage
        if coverage + 1e-12 < IMAGE_FRACTION:
            candidate['reason'] = 'image_coverage'
            counts['image_coverage'] += 1
            candidates.append(candidate)
            continue
        patch = rgb.crop(box)
        mask = np.asarray(Image.fromarray(full_mask).crop(box))
        pixel_counts = np.bincount(mask.ravel(), minlength=3)
        if len(pixel_counts) != 3 or not (pixel_counts[1] or pixel_counts[2]):
            raise ValueError(f'Unexpected/empty mask: {image_id}:{index}')
        filename = f'{Path(image_info["file_name"]).stem}_{index}.png'
        image_sha, image_bytes = save_png(output / 'img' / filename, patch)
        mask_sha, mask_bytes = save_png(output / 'mask' / filename, Image.fromarray(mask))
        histogram += np.asarray(patch.histogram(), np.int64).reshape(3, 256)
        counts['keep'] += 1
        kind = 'both' if pixel_counts[1] and pixel_counts[2] else 'mitotic_only' if pixel_counts[1] else 'non_mitotic_only'
        classes[kind] += 1
        patches.append({'filename': filename, 'image_id': image_id, 'source_filename': image_info['file_name'],
                        'tumor_type': image_info['tumor_type'], 'split': image_info['split'], 'grid_index': index,
                        'left': left, 'top': top, 'right': right, 'bottom': bottom,
                        'annotation_ids': [a['id'] for a in objects], 'min_object_visibility': min(fractions),
                        'valid_image_fraction': coverage, 'class_pixels': pixel_counts.tolist(),
                        'image_sha256': image_sha, 'mask_sha256': mask_sha,
                        'image_bytes': image_bytes, 'mask_bytes': mask_bytes, 'patch_class': kind})
        candidates.append(candidate)
    rgb.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during extraction: {path}')
    return {'image_id': image_id, 'source_filename': image_info['file_name'], 'split': image_info['split'],
            'source_sha256': raw_sha, 'source_bytes': after.st_size, 'source_mtime_ns': after.st_mtime_ns,
            'counts': dict(counts), 'patch_classes': dict(classes), 'patches': patches,
            'candidates': candidates, 'rgb_histogram': histogram.tolist(),
            'elapsed_seconds': round(time.monotonic() - began, 3)}


def check_receipt(output, receipt):
    for row in receipt['patches']:
        for folder, key in [('img', 'image_sha256'), ('mask', 'mask_sha256')]:
            path = output / folder / row['filename']
            if not path.is_file() or digest_file(path) != row[key]:
                raise ValueError(f'Completed output missing or modified: {path}')


def channel_stats(histogram):
    h = np.asarray(histogram, np.float64)
    if not h.sum():
        return None
    x = np.arange(256, dtype=np.float64) / 255
    means = (h * x).sum(axis=1) / h.sum(axis=1)
    variances = (h * x**2).sum(axis=1) / h.sum(axis=1) - means**2
    return {'mean': means.tolist(), 'std': np.sqrt(np.maximum(0, variances)).tolist(),
            'pixel_count_per_channel': int(h[0].sum())}


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                             for k, v in row.items() if k in fields})


def finalize(output, binding, by_image):
    receipts = [json.loads((output / 'metadata/slides' / f'{im["id"]:06}.json').read_text()) for im in binding['images']]
    patches = [p for receipt in receipts for p in receipt['patches']]
    names = {p['filename'] for p in patches}
    if len(names) != len(patches):
        raise ValueError('Duplicate patch names')
    for folder in ['img', 'mask']:
        actual = {p.name for p in (output / folder).iterdir()}
        if actual != names:
            raise ValueError(f'Unexpected/missing output files in {folder}: {actual ^ names}')
    totals, classes, split_counts = Counter(), Counter(), Counter()
    hist = np.zeros((3, 256), np.int64)
    train_hist = np.zeros((3, 256), np.int64)
    for receipt in receipts:
        totals.update(receipt['counts'])
        classes.update(receipt['patch_classes'])
        split_counts[receipt['split']] += len(receipt['patches'])
        hist += receipt['rgb_histogram']
        if receipt['split'] == 'train':
            train_hist += receipt['rgb_histogram']
    fields = list(patches[0])
    write_csv(output / 'patch_manifest.csv', patches, fields)
    (output / 'splits').mkdir(exist_ok=True)
    for split in ['train', 'test']:
        write_csv(output / 'splits' / f'{split}.csv', [p for p in patches if p['split'] == split], fields)
    write_csv(output / 'image_groups.csv', binding['images'], ['id', 'file_name', 'tumor_type', 'split', 'width', 'height'])
    occurrences = Counter(a for row in patches for a in row['annotation_ids'])
    coverage = [{'annotation_id': a['id'], 'image_id': a['image_id'], 'category_id': a['category_id'],
                 'retained_patch_occurrences': occurrences[a['id']]} for anns in by_image.values() for a in anns]
    write_csv(output / 'annotation_coverage.csv', coverage, list(coverage[0]))
    write_csv(output / 'candidate_decisions.csv',
              ({'image_id': r['image_id'], **c} for r in receipts for c in r['candidates']),
              ['image_id', 'grid_index', 'box', 'reason', 'valid_image_fraction'])
    summary = {'status': 'built_pending_independent_validation', 'source_images': len(receipts),
               'patch_pairs': len(patches), 'counts': dict(totals), 'patch_classes': dict(classes),
               'patches_by_upstream_split': dict(split_counts),
               'source_annotations': len(coverage), 'annotations_represented': len(occurrences),
               'annotations_unrepresented': sum(r['retained_patch_occurrences'] == 0 for r in coverage),
               'unrepresented_annotations_by_class': dict(Counter(r['category_id'] for r in coverage if not r['retained_patch_occurrences'])),
               'image_and_mask_bytes': sum(p['image_bytes'] + p['mask_bytes'] for p in patches),
               'normalization_all_patches': channel_stats(hist), 'normalization_upstream_train_only': channel_stats(train_hist),
               'recipe': RECIPE, 'annotation_sha256': binding['annotation_sha256'],
               'builder_sha256': binding['builder_sha256'],
               'manifest_sha256': digest_file(output / 'patch_manifest.csv')}
    atomic_json(output / 'build_summary.json', summary)
    return summary


def build(args, binding, by_image):
    output = args.output.resolve()
    source = args.source.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from the raw dataset tree')
    if output.exists() and any(output.iterdir()) and not (output / 'build_config.json').is_file():
        raise ValueError('Refusing to use a nonempty, unrecognized output directory')
    if shutil.disk_usage(output.parent if output.parent.exists() else source.parent).free < 50 * 2**30:
        raise RuntimeError('Less than 50 GiB free; refusing dataset build')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / 'build_config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != binding:
                raise ValueError('Recipe/source/code differs from the existing build; do not mix versions')
        else:
            atomic_json(config_path, binding)
        for folder in ['img', 'mask', 'metadata/slides']:
            (output / folder).mkdir(parents=True, exist_ok=True)
        selected = set(args.image_ids or [i['id'] for i in binding['images']])
        if selected - {i['id'] for i in binding['images']}:
            raise ValueError('Unknown requested image IDs')
        completed = set()
        for im in binding['images']:
            receipt_path = output / 'metadata/slides' / f'{im["id"]:06}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                source_stat = (source / 'images' / im['file_name']).stat()
                if (source_stat.st_size, source_stat.st_mtime_ns) != (receipt['source_bytes'], receipt['source_mtime_ns']):
                    raise ValueError('Completed source image changed; use a new output')
                check_receipt(output, receipt)
                completed.add(im['id'])
        for im in binding['images']:
            if im['id'] not in selected or im['id'] in completed:
                continue
            if shutil.disk_usage(output).free < 50 * 2**30:
                raise RuntimeError('Disk free reserve reached')
            receipt = process_image(source, output, im, by_image[im['id']])
            atomic_json(output / 'metadata/slides' / f'{im["id"]:06}.json', receipt)
            completed.add(im['id'])
            status = {'status': 'building', 'completed_images': len(completed), 'total_images': len(binding['images']),
                      'last_image': im['file_name'], 'last_patch_pairs': len(receipt['patches']),
                      'last_elapsed_seconds': receipt['elapsed_seconds'], 'updated_unix': time.time()}
            atomic_json(output / 'progress.json', status)
            print(json.dumps(status), flush=True)
        if len(completed) == len(binding['images']):
            summary = finalize(output, binding, by_image)
            atomic_json(output / 'progress.json', {'status': summary['status'], 'completed_images': len(completed),
                                                  'total_images': len(completed), 'patch_pairs': summary['patch_pairs']})
            print(json.dumps(summary, indent=2), flush=True)
        else:
            print(json.dumps({'status': 'partial_build', 'completed_images': len(completed), 'total_images': len(binding['images'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'build'])
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--split-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image-ids', nargs='+', type=int, help='Process these images now; keep the full build binding for resumable pilot runs')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    binding, by_image = load_sources(args.source, args.split_csv)
    if args.mode == 'plan':
        report = make_plan(binding, by_image)
        atomic_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != 'binding'}, indent=2))
    else:
        build(args, binding, by_image)


if __name__ == '__main__':
    main()
