#!/usr/bin/env bash
# Launch the FoundationPose + SAM2 playground container on the RTX 4060 laptop (Ada, sm_89).
# Same image recipe as run_container_blackwell.sh, built for sm_89:
#   docker build -f docker/Dockerfile.blackwell --build-arg TORCH_CUDA_ARCH_LIST=8.9 -t fp-sam2:ada .
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
  fp-sam2:ada bash
