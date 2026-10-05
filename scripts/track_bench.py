"""Step 0 of realtime_fp.md: how fast is tracking on this GPU, with no network at all?

    python scripts/track_bench.py                    # inside the container
    python scripts/track_bench.py --n 300 --iters 1 2 --margin 40

Registers once on a saved frame (Data/Input/{rgb,depth}.png + camera.json, with the
mask of the last /predict_pose), then runs the tracking server's own per-frame step
(`Tracker.step`: depth -> metres, track_one, fit) N times per configuration on that
same still frame, on the full frame and on the ROI crop the server would ask for.

Reports per configuration: ms/frame (mean, p50, p95) and the rate it implies, peak
GPU memory, the fit, and the pose jitter (the part is still, so any motion is
noise). Also the payload sizes and encode times of S5, so the network budget of
step 1 can be checked against real numbers.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import trimesh

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "scripts"))

import torch  # noqa: E402
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr, set_logging_format, set_seed  # noqa: E402
import fp_stream as fs  # noqa: E402
from fp_tracking import Tracker, register_cropped  # noqa: E402

RESULTS = os.path.join(CODE_DIR, "Data", "Output", "foundationpose_results")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--frame-dir", default=os.path.join(CODE_DIR, "Data", "Input"))
    p.add_argument("--mask", default=os.path.join(RESULTS, "mask.png"))
    p.add_argument("--object", default=None,
                   help="CAD stem (default: obj_name of the last detection_pem.json)")
    p.add_argument("--cad-dir", default=os.path.join(CODE_DIR, "Data", "Input"))
    p.add_argument("--mesh-scale", type=float, default=0.001)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, nargs="+", default=[1, 2])
    p.add_argument("--margin", type=int, default=40, help="ROI motion margin, px")
    p.add_argument("--register-iter", type=int, default=5)
    return p.parse_args()


def default_object():
    try:
        with open(os.path.join(RESULTS, "detection_pem.json")) as fh:
            return json.load(fh)[0]["obj_name"]
    except (OSError, KeyError, IndexError, ValueError):
        return "test_objv2_ear"


def load_frame(frame_dir):
    with open(os.path.join(frame_dir, "camera.json")) as fh:
        cam = json.load(fh)
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    scale_m = float(cam.get("depth_scale", 1.0)) / 1000.0  # raw -> mm -> m
    rgb = cv2.cvtColor(cv2.imread(os.path.join(frame_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(frame_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise SystemExit("depth.png must be a 16-bit PNG")
    return K, scale_m, rgb, depth


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def jitter(poses):
    """Still part: spread of the tracked poses. Translation as the RMS distance from
    the mean (mm), rotation as the RMS angle from the first pose (deg)."""
    P = np.asarray(poses)
    t = P[:, :3, 3]
    t_rms = float(np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()) * 1000.0)
    angles = [rot_angle_deg(P[0, :3, :3], p[:3, :3]) for p in P]
    r_rms = float(np.sqrt(np.mean(np.square(angles))))
    return t_rms, r_rms


def payload_report(rgb, depth, roi):
    out = {}
    for name, r in (("full", None), ("roi", roi)):
        c_rgb, c_d = fs.crop(rgb, r), fs.crop(depth, r)
        t = time.perf_counter()
        jb = fs.encode_rgb(c_rgb, "jpeg", 90)
        t_j = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        pb = fs.encode_depth(c_d, "png")
        t_p = (time.perf_counter() - t) * 1e3
        raw = c_rgb.nbytes + c_d.nbytes
        out[name] = (c_rgb.shape[1], c_rgb.shape[0], len(jb) + len(pb), raw, t_j + t_p)
    return out


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)
    logging.getLogger().setLevel(logging.WARNING)  # estimater is chatty

    name = args.object or default_object()
    cad = os.path.join(args.cad_dir, f"{name}.ply")
    mesh = trimesh.load(cad, force="mesh")
    mesh.apply_scale(args.mesh_scale)
    K, scale_m, rgb, depth_raw = load_frame(args.frame_dir)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != depth_raw.shape:
        raise SystemExit(f"need a mask the size of the frame at {args.mask}")
    H, W = depth_raw.shape
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu} | CAD {name} ({len(mesh.faces)} faces, extents "
          f"{np.round(mesh.extents * 1000).astype(int)} mm) | frame {W}x{H}")

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         glctx=dr.RasterizeCudaContext(), debug=0,
                         debug_dir=os.path.join(CODE_DIR, "debug"))
    est._loaded_cad = cad

    depth_m = depth_raw.astype(np.float32) * scale_m
    depth_m[(depth_m < 0.001) | (depth_m >= 3.0)] = 0
    t = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pose0 = register_cropped(est, K, rgb, depth_m, mask > 0, iteration=args.register_iter)
    torch.cuda.synchronize()
    print(f"register (cropped): {time.perf_counter() - t:.2f} s (cold; includes warm-up), "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, "
          f"t = {np.round(pose0[:3, 3], 4)} m")

    tracker = Tracker(est, threading.Lock(), zfar=3.0)
    hint = fs.projection_hint(est.pose_last.cpu().numpy(), K, est.diameter)
    roi = fs.roi_from_hint(hint, (H, W), margin_px=args.margin)

    rows = []
    for it in args.iters:
        for label, r in (("full", None), ("roi", roi)):
            tracker.start({"seed": "pose", "pose": pose0.reshape(-1).tolist(),
                           "refine_iter": it, "ego_motion": False, "auto_reseed": False})
            header = {"K": K.reshape(-1).tolist(), "full_hw": [H, W], "roi": r,
                      "depth": {"scale": scale_m}, "T_base_cam": None}
            c_rgb, c_d = fs.crop(rgb, r), fs.crop(depth_raw, r)
            torch.cuda.reset_peak_memory_stats()
            times, poses, fits, stage = [], [], [], {}
            for i in range(args.warmup + args.n):
                t = time.perf_counter()
                reply, tm = tracker.step(header, c_rgb, c_d)
                dt = (time.perf_counter() - t) * 1e3
                if i < args.warmup:
                    continue
                times.append(dt)
                fits.append(reply["fit"])
                if reply["pose"] is not None:
                    poses.append(np.asarray(reply["pose"]))
                for k, v in tm.items():
                    stage[k] = stage.get(k, 0.0) + v
            times = np.asarray(times)
            t_rms, r_rms = jitter(poses) if poses else (float("nan"),) * 2
            drift = (np.linalg.norm(poses[-1][:3, 3] - pose0[:3, 3]) * 1000.0
                     if poses else float("nan"))
            rows.append({
                "iter": it, "input": label,
                "size": f"{c_rgb.shape[1]}x{c_rgb.shape[0]}",
                "mean": times.mean(), "p50": np.percentile(times, 50),
                "p95": np.percentile(times, 95), "hz": 1000.0 / times.mean(),
                "mem": torch.cuda.max_memory_allocated() / 2**20,
                "fit": float(np.mean(fits)), "t_rms": t_rms, "r_rms": r_rms,
                "drift": drift, "state": tracker.state,
                "stages": {k: v / args.n for k, v in stage.items()},
            })

    print(f"\ntracking step, {args.n} frames each (still part; warm-up {args.warmup} skipped)")
    print(f"{'iter':>4} {'input':>5} {'size':>9} {'mean':>7} {'p50':>7} {'p95':>7} "
          f"{'Hz':>6} {'GPU MB':>7} {'fit':>5} {'jit mm':>7} {'jit deg':>7} {'drift mm':>8}")
    for r in rows:
        print(f"{r['iter']:>4} {r['input']:>5} {r['size']:>9} {r['mean']:7.1f} "
              f"{r['p50']:7.1f} {r['p95']:7.1f} {r['hz']:6.1f} {r['mem']:7.0f} "
              f"{r['fit']:5.2f} {r['t_rms']:7.2f} {r['r_rms']:7.3f} {r['drift']:8.2f}")
    print("\nper-stage mean, ms:")
    for r in rows:
        st = "  ".join(f"{k.replace('_ms', '')}={v:.1f}" for k, v in r["stages"].items())
        print(f"  iter={r['iter']} {r['input']:>4}: {st}")

    print(f"\npayload per frame (rgb JPEG q90 + depth PNG lvl1), ROI margin {args.margin}px:")
    for label, (w, h, enc, raw, ms) in payload_report(rgb, depth_raw, roi).items():
        print(f"  {label:>4} {w}x{h}: {enc / 1024:7.1f} KB encoded ({raw / 1024:7.1f} KB raw), "
              f"encode {ms:.1f} ms")


if __name__ == "__main__":
    main()
