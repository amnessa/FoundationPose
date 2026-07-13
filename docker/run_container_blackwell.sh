#!/usr/bin/env bash
# Launch the FoundationPose + SAM2 playground container.
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
NAME=fp-sam2

docker rm -f "${NAME}" 2>/dev/null || true

# Let the container draw to the host X server (cv2.imshow / open3d debug windows).
xhost +local:root >/dev/null 2>&1 || true

docker run --gpus all -it \
  --name "${NAME}" \
  --network=host \
  --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e DISPLAY="${DISPLAY:-}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${PROJ_ROOT}:/workspace/FoundationPose" \
  -v "${HOME}/.cache/torch_extensions:/root/.cache/torch_extensions" \
  -w /workspace/FoundationPose \
  fp-sam2:latest bash
