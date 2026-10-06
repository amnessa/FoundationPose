"""Tracking-only FoundationPose server, for the laptop (RTX 4060, 8 GB).

Registration happens on the desktop (fp_server.py /predict_pose: SAM2, PPF and the
scorer, full frame). This process only follows a part from a pose it is given, so it
loads just the refiner network (~300 MB of GPU memory in total) and never
registers: a full-frame register does not fit 8 GB.

    laptop ROS node ── ws://127.0.0.1:5001 ──► this process (laptop container)
        │  seed: {"type":"start","seed":"pose","pose":[16],"object":"<cad stem>",
        │         "T_base_cam":[16]}  -- the desktop's /predict_pose answer
        │  frames: fp_stream.pack_frame(...), full frame or ROI
        └─◄ pose per frame (camera frame, metres), fit, state

When the reply says LOST, the client re-seeds through the desktop: it sets
`want_reseed_mask` on a frame, gets back the CAD silhouette at the last good pose,
POSTs that frame + mask to /predict_pose, and sends a new `start` with the answer.
Meanwhile this process keeps trying a cheap track_one from the last good pose, and
recovers on its own if the part was only briefly covered.

Protocol: fp_stream.py. Tracking state machine: fp_tracking.Tracker.

    python scripts/fp_track_server.py            # inside docker/run_container.sh

Configuration (env vars, all optional):

    CAD_DIR      where `<object>.ply` seeds are looked up        (Data/Input)
    MESH_SCALE   multiplied into the mesh on load (mm -> m)      (0.001)
    MESH_PATH    CAD loaded at startup, until a seed names another
                                                    (Data/Input/test_objv2_ear.ply)
    TRACK_HOST   bind address; the client is on this machine    (127.0.0.1)
    TRACK_PORT                                                    (5001)
    ZFAR         depth beyond this many metres is discarded      (3.0)
"""

import logging
import os
import sys
import threading
import time

import trimesh

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "scripts"))

import torch  # noqa: E402
from estimater import FoundationPose, PoseRefinePredictor, dr, set_logging_format, set_seed  # noqa: E402
from fp_tracking import Tracker, TrackingServer  # noqa: E402

DATA_DIR = os.path.join(CODE_DIR, "Data", "Input")
CAD_DIR = os.environ.get("CAD_DIR", DATA_DIR)
MESH_SCALE = float(os.environ.get("MESH_SCALE", "0.001"))
MESH_PATH = os.environ.get("MESH_PATH", os.path.join(DATA_DIR, "test_objv2_ear.ply"))
TRACK_HOST = os.environ.get("TRACK_HOST", "127.0.0.1")
TRACK_PORT = int(os.environ.get("TRACK_PORT", "5001"))
ZFAR = float(os.environ.get("ZFAR", "3.0"))


class _NoScorer:
    """Stands in for the scorer network so FoundationPose() does not load it: the
    scorer only ranks register() hypotheses, and this process never registers."""

    def predict(self, *args, **kwargs):
        raise RuntimeError("tracking-only server: register on the desktop's /predict_pose")


_meshes = {}


def load_cad(path):
    if path not in _meshes:
        mesh = trimesh.load(path, force="mesh")
        if MESH_SCALE != 1.0:
            mesh.apply_scale(MESH_SCALE)
        _meshes[path] = mesh
        logging.info(f"loaded CAD {os.path.basename(path)}: "
                     f"extents={mesh.extents.round(4)} m, {len(mesh.faces)} faces")
    return _meshes[path]


def main():
    set_logging_format()
    set_seed(0)
    started = time.perf_counter()

    mesh = load_cad(MESH_PATH)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=_NoScorer(), refiner=PoseRefinePredictor(),
                         glctx=dr.RasterizeCudaContext(), debug=0,
                         debug_dir=os.path.join(CODE_DIR, "debug"))
    est._loaded_cad = MESH_PATH

    def use_cad(name):
        """A seed's object name (the desktop's PPF answer, a .ply stem) -> EST's CAD."""
        path = os.path.join(CAD_DIR, f"{name}.ply")
        if not os.path.exists(path):
            raise ValueError(f"no CAD named {name!r} in CAD_DIR={CAD_DIR}; copy the "
                             "desktop's Data/Input/*.ply here")
        if est._loaded_cad != path:
            m = load_cad(path)
            est.reset_object(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m)
            est._loaded_cad = path

    tracker = Tracker(est, threading.Lock(), use_cad=use_cad, can_register=False, zfar=ZFAR)
    server = TrackingServer(tracker, host=TRACK_HOST, port=TRACK_PORT)
    logging.info(f"tracking-only server ready in {time.perf_counter() - started:.1f} s, "
                 f"GPU {torch.cuda.get_device_name(0)}, "
                 f"{torch.cuda.memory_allocated() / 2**20:.0f} MB allocated")
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("stopped")


if __name__ == "__main__":
    main()
