#!/usr/bin/env python3
"""Validate preservation and publication boundaries without model dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'.ckpt', '.pt', '.pth', '.safetensors', '.onnx', '.tar', '.gz', '.zip',
           '.whl', '.tif', '.tiff', '.svs', '.ndpi', '.mha', '.png', '.jpg',
           '.jpeg', '.npy', '.npz', '.h5', '.sqlite', '.db', '.parquet', '.pdf'}
SKIP = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'venv'}


def main():
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts)]
    manifest = json.loads((ROOT / 'docs/source_manifest.json').read_text())
    seen = set()
    for item in manifest:
        name = item['path']
        if name in seen:
            errors.append(f'Duplicate manifest entry: {name}')
        seen.add(name)
        p = ROOT / name
        if not p.is_file():
            errors.append(f'Missing preserved file: {name}')
        elif hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f'Changed preserved file: {name}')
    python_count = shell_count = 0
    for p in files:
        name = str(p.relative_to(ROOT))
        if p.is_symlink() or p.suffix.lower() in BLOCKED or p.stat().st_size > 5 * 1024**2:
            errors.append(f'Excluded artifact in publication: {name}')
        try:
            text = p.read_text(encoding='utf-8')
            if re.search(r'\bgh[pousr]_[A-Za-z0-9_]{20,}\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{20,}\b', text):
                errors.append(f'Credential pattern: {name}')
            if p.suffix == '.py':
                ast.parse(text, filename=name)
                python_count += 1
            elif p.suffix == '.json':
                json.loads(text)
            elif p.suffix == '.sh':
                result = subprocess.run(['bash', '-n', str(p)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(f'Shell syntax: {name}: {result.stderr.strip()}')
                shell_count += 1
        except (UnicodeDecodeError, SyntaxError, json.JSONDecodeError) as exc:
            errors.append(f'{name}: {exc}')
    for variant in ['midog', 'midog_v2']:
        context = ROOT / 'docker' / variant
        for name in ['Dockerfile', 'algorithm/loader.py', 'algorithm/network/ASPP.py',
                     'algorithm/network/DeepLabv3_plus.py', 'inference.py', 'entrypoint.sh',
                     'requirements.inference.txt', 'resources/patch_config.yaml',
                     'resources/model_config.yaml', 'resources/inference_config.yaml']:
            if not (context / name).is_file():
                errors.append(f'Missing Docker build source: {variant}/{name}')
    print(json.dumps({'files': len(files), 'preserved_files': len(manifest), 'python_files': python_count,
                      'shell_files': shell_count, 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
