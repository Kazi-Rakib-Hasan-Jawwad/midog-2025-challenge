#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR="${1:-/input}"
OUTPUT_DIR="${2:-/output}"
mkdir -p "$OUTPUT_DIR"
python -m inference --input "$INPUT_DIR" --output "$OUTPUT_DIR"
