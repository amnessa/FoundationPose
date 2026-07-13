#!/usr/bin/env bash
# Download SAM2.1 checkpoints into the mounted repo (weights/sam2/) so they survive
# image rebuilds. /opt/sam2/checkpoints lives inside the image and would be lost.
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
DEST="${PROJ_ROOT}/weights/sam2"
BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Default to the small model; pass e.g. `large` for the best quality.
MODEL="${1:-small}"
case "${MODEL}" in
  tiny)      FILE=sam2.1_hiera_tiny.pt;      CFG=sam2.1_hiera_t.yaml ;;
  small)     FILE=sam2.1_hiera_small.pt;     CFG=sam2.1_hiera_s.yaml ;;
  base_plus) FILE=sam2.1_hiera_base_plus.pt; CFG=sam2.1_hiera_b+.yaml ;;
  large)     FILE=sam2.1_hiera_large.pt;     CFG=sam2.1_hiera_l.yaml ;;
  *) echo "usage: $0 [tiny|small|base_plus|large]" >&2; exit 1 ;;
esac

mkdir -p "${DEST}"
echo "Downloading ${FILE} -> ${DEST}"
wget -c -O "${DEST}/${FILE}" "${BASE}/${FILE}"
echo "Done: ${DEST}/${FILE}"
echo "Matching config (pass to build_sam2): configs/sam2.1/${CFG}"
