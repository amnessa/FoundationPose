# FoundationPose + SAM2 on RTX 50-series (Blackwell)

The upstream `docker/dockerfile` targets `cudagl:11.3` with PyTorch 2.0/cu118. That stack has
no `sm_120` kernels, so it **cannot run on a Blackwell GPU** (RTX 5070 Ti / 5080 / 5090) —
you'd get `no kernel image is available for execution on the device`.

`Dockerfile.blackwell` replaces it:

| | upstream | this image |
|---|---|---|
| CUDA | 11.3 | 12.8 |
| PyTorch | 2.0 / cu118 | 2.8 / cu128 |
| arch | ≤ sm_86 | `sm_120` (`TORCH_CUDA_ARCH_LIST=12.0`) |
| Python | 3.8 (conda) | 3.10 (system) |
| kaolin | built | omitted (see below) |
| SAM2 | — | `/opt/sam2`, editable install |

Conda is dropped: mixing conda's `cxx-compiler` toolchain with the image's `nvcc` is a
common source of ABI breakage, and nothing here needs it.

**Kaolin is intentionally omitted.** It is imported lazily in `Utils.py` (`OctreeManager`,
line ~908) and is only reached by the *model-free / NeRF* path. Model-based pose estimation
and tracking — `run_demo.py`, `run_ycb_video.py`, `run_linemod.py` — never touch it. Kaolin
is also the single hardest dep to build against a recent torch, so it's not worth carrying
until you actually need the model-free path.

## Build

```bash
docker build -f docker/Dockerfile.blackwell -t fp-sam2:latest .
```

Expect 30–50 min, almost all of it in the pytorch3d source build (there are no prebuilt
pytorch3d wheels for torch 2.8 / cu128).

## Run

```bash
bash docker/run_container_blackwell.sh
```

Mounts the repo at `/workspace/FoundationPose` and shares the host X socket, so `cv2.imshow`
and Open3D debug windows work. It also mounts `~/.cache/torch_extensions` so nvdiffrast's
JIT-compiled CUDA kernels are built once, not on every container start.

## First run, inside the container

```bash
bash docker/build_extensions.sh   # builds mycpp, then runs check_env.py
```

`check_env.py` should report torch seeing the 5070 Ti, plus `pytorch3d`, `nvdiffrast.torch`,
`trimesh`, `open3d`, `cv2` and `warp` all importing.

## Weights and data (not in the image)

FoundationPose weights and demo data are Google Drive downloads — see the repo readme.
Put them in `weights/` and `demo_data/`; both are inside the mounted repo, so they persist.

```bash
python run_demo.py   # after weights/ and demo_data/mustard0 exist
```

Note: the first nvdiffrast call compiles CUDA kernels for `sm_120` and takes a minute or two.
That's a one-time cost, cached in the mounted `torch_extensions` dir.

## SAM2

Installed editable at `/opt/sam2`, so `import sam2` works anywhere and you can edit its source.

```bash
bash docker/download_sam2_ckpts.sh small   # -> weights/sam2/  (tiny|small|base_plus|large)
```

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",
                  "weights/sam2/sam2.1_hiera_small.pt", device="cuda")
predictor = SAM2ImagePredictor(sam2)
```

The image sets `SAM2_BUILD_CUDA=0`. SAM2's optional CUDA extension only powers a mask
post-processing refinement step; upstream marks it optional and the model runs fine without
it. Building it against cu128 is avoidable friction.

The obvious thing to wire up: SAM2 produces the object mask that FoundationPose's
`est.register(..., ob_mask=...)` needs on frame 0, replacing the demo's hand-drawn mask.
