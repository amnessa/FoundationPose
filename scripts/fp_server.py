"""FoundationPose pose-initialization server.

Drop-in replacement for the SAM-6D `server.py` this project used to talk to.
The client (see `foundationpose_bridge_node.py`) POSTs the same three files it
always did -- rgb.png, depth.png, camera.json -- and gets back a single 6D pose.

Differences from the SAM-6D server, all of them consequences of what
FoundationPose actually is:

- **It has no detector.** SAM-6D segmented the scene itself (the ISM stage);
  FoundationPose needs to be *told* which pixels are the object. So on each
  request we open the received frame in a window, you click the object, SAM2
  turns the click into a mask, and that mask seeds `est.register()`. Send a
  `mask` file or a `click` field instead to skip the window entirely.
- **The model stays resident.** SAM-6D re-ran `demo.sh` (template rendering ->
  segmentation -> pose) as a subprocess on every request. Here the mesh, the
  refiner/scorer networks and SAM2 are loaded once at startup, so a request is
  a registration and nothing else.
- **One object, one pose.** SAM-6D returned a ranked list of detections; this
  returns the single pose of the thing you clicked, as a 4x4 matrix.
- **Metres, not millimetres.** The CAD is scaled to metres on load (BOP-style
  .ply files are mm), so the translation comes back in metres. The client must
  not divide by 1000 anymore.

Configuration (all optional, via env vars):

    MESH_PATH        CAD model of the object          (Data/Input/test_objv2_base.ply)
    MESH_SCALE       multiplied into the mesh on load (0.001 -- i.e. mm -> m)
    SYMMETRY_INFO    BOP models_info-style .json with `symmetries_discrete` /
                     `symmetries_continuous` for the object. Strongly recommended
                     for symmetric parts (plates, blocks): without it, registration
                     picks an arbitrary member of the symmetry group and the pose
                     appears to flip between otherwise-identical runs.
    EST_REFINE_ITER  refinement iterations in register()  (5)
    ZFAR             depth beyond this many metres is discarded (3.0)
    SAM2_CKPT/CFG    SAM2 checkpoint and its matching config
    PORT             (5000)
"""

import os
import sys
import json
import time
import base64
import logging
import threading

import cv2
import numpy as np
import trimesh
from flask import Flask, request, jsonify

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from estimater import (  # noqa: E402
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    set_logging_format,
    set_seed,
)
from Utils import draw_posed_3d_box, draw_xyz_axis, symmetry_tfs_from_info  # noqa: E402

app = Flask(__name__)

DATA_DIR = os.path.join(CODE_DIR, "Data", "Input")
OUTPUT_DIR = os.path.join(CODE_DIR, "Data", "Output", "foundationpose_results")

MESH_PATH = os.environ.get("MESH_PATH", os.path.join(DATA_DIR, "test_objv2_base.ply"))
MESH_SCALE = float(os.environ.get("MESH_SCALE", "0.001"))
SYMMETRY_INFO = os.environ.get("SYMMETRY_INFO", "")
EST_REFINE_ITER = int(os.environ.get("EST_REFINE_ITER", "5"))
ZFAR = float(os.environ.get("ZFAR", "3.0"))

SAM2_CKPT = os.environ.get(
    "SAM2_CKPT", os.path.join(CODE_DIR, "weights", "sam2", "sam2.1_hiera_small.pt"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")

PORT = int(os.environ.get("PORT", "5000"))

# In Docker we run as root on a bind-mounted workspace, so anything we write is
# root-owned on the host and the host user can't overwrite it afterwards.
# run_container.sh passes the host identity in; hand ownership back. (Same trick
# the SAM-6D server used.)
HOST_UID = os.environ.get("HOST_UID")
HOST_GID = os.environ.get("HOST_GID")

# register() is GPU-bound and the click window is modal; two at once makes no
# sense. Reject rather than queue, so a repeatedly-triggering client fails fast.
_lock = threading.Lock()

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _give_back_ownership(path):
    if not HOST_UID or os.geteuid() != 0:
        return
    try:
        os.chown(path, int(HOST_UID), int(HOST_GID or HOST_UID))
    except OSError as err:
        logging.warning(f"could not chown {path} back to the host user: {err}")


# ── Model loading (once, at startup) ──────────────────────────────────────

def load_estimator():
    mesh = trimesh.load(MESH_PATH, force='mesh')
    if MESH_SCALE != 1.0:
        mesh.apply_scale(MESH_SCALE)

    # FoundationPose renders the mesh to compare against the RGB crop. An
    # untextured CAD model is fine -- make_mesh_tensors() falls back to flat gray
    # vertex colours -- registration then leans on geometry/depth, which is the
    # dominant signal anyway. Nothing to do here but let it happen.
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        tex = "uv texture"
    elif getattr(mesh.visual, 'vertex_colors', None) is not None:
        tex = "vertex colours"
    else:
        tex = "none (flat gray)"

    symmetry_tfs = None
    if SYMMETRY_INFO:
        with open(SYMMETRY_INFO) as fh:
            symmetry_tfs = symmetry_tfs_from_info(json.load(fh))

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        symmetry_tfs=symmetry_tfs,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir=os.path.join(CODE_DIR, "debug"),
    )
    logging.info(
        f"mesh={MESH_PATH} scale={MESH_SCALE} extents={mesh.extents.round(4)}m "
        f"texture={tex} symmetry_tfs={0 if symmetry_tfs is None else len(symmetry_tfs)}")
    return est, mesh


def load_sam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda'))
    logging.info(f"SAM2 ready: {SAM2_CKPT}")
    return predictor


# ── Request payload → numpy ───────────────────────────────────────────────

def read_camera(camera_path):
    """camera.json is the SAM-6D/BOP payload the client already sends:
    {"cam_K": [9 floats], "depth_scale": <raw units -> mm>}."""
    with open(camera_path) as fh:
        payload = json.load(fh)
    K = np.array(payload['cam_K'], dtype=np.float64).reshape(3, 3)
    depth_scale = float(payload.get('depth_scale', 1.0))
    return K, depth_scale


def read_rgb(rgb_path):
    # FoundationPose and SAM2 both want RGB; cv2 hands us BGR.
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode rgb image at {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth(depth_path, depth_scale):
    """16-bit PNG in raw sensor units -> float32 metres, which is what
    FoundationPose expects everywhere (and why the mesh had to be metres too)."""
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not decode depth image at {depth_path}")
    depth = raw.astype(np.float32) * depth_scale / 1000.0
    depth[(depth < 0.001) | (depth >= ZFAR) | ~np.isfinite(depth)] = 0
    return depth


# ── Mask: SAM2, driven by clicks on the received frame ────────────────────

def sam2_mask(predictor, rgb, points, labels):
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points, dtype=np.float32),
        point_labels=np.array(labels, dtype=np.int32),
        multimask_output=len(points) == 1,  # one click is ambiguous; let it propose
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def click_for_mask(predictor, rgb):
    """Show the frame, let the operator click the object, return a boolean mask.

    Left click adds a positive point, right click a negative one (to carve away
    a neighbouring part SAM2 grabbed by mistake). The mask is recomputed after
    every click so you see what you are about to register against.
    """
    predictor.set_image(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    win = "click the object  [L=object  R=not-object  u=undo  r=reset  ENTER=accept  ESC=abort]"

    state = {'points': [], 'labels': [], 'mask': None, 'score': 0.0, 'dirty': False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['points'].append([x, y])
            state['labels'].append(1)
            state['dirty'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            state['points'].append([x, y])
            state['labels'].append(0)
            state['dirty'] = True

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            if state['dirty']:
                state['dirty'] = False
                if state['points']:
                    state['mask'], state['score'] = sam2_mask(
                        predictor, rgb, state['points'], state['labels'])
                else:
                    state['mask'] = None

            vis = bgr.copy()
            if state['mask'] is not None:
                overlay = vis.copy()
                overlay[state['mask']] = (0, 255, 0)
                vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
                contours, _ = cv2.findContours(
                    state['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            for (x, y), label in zip(state['points'], state['labels']):
                cv2.circle(vis, (int(x), int(y)), 5, (0, 200, 0) if label else (0, 0, 255), -1)
                cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)

            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # ENTER
                if state['mask'] is not None and state['mask'].any():
                    return state['mask'], state['score']
                logging.warning("nothing selected yet -- click the object first")
            elif key == 27:  # ESC
                return None, 0.0
            elif key == ord('r'):
                state['points'].clear()
                state['labels'].clear()
                state['mask'] = None
            elif key == ord('u') and state['points']:
                state['points'].pop()
                state['labels'].pop()
                state['dirty'] = True
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)


def resolve_mask(predictor, rgb):
    """Mask, in order of preference: one uploaded by the client, a click point
    the client sent, or the interactive window."""
    if 'mask' in request.files:
        buf = np.frombuffer(request.files['mask'].read(), np.uint8)
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("could not decode the uploaded mask")
        return mask > 0, 1.0, "uploaded"

    click = request.form.get('click')
    if click:
        spec = json.loads(click)
        points = spec if isinstance(spec, list) else [[spec['u'], spec['v']]]
        labels = [1] * len(points)
        mask, score = sam2_mask(predictor, rgb, points, labels)
        return mask, score, "sam2(client click)"

    if not os.environ.get('DISPLAY'):
        raise RuntimeError(
            "no mask, no click, and no DISPLAY to open the click window in. "
            "Run the container with X11 forwarding (docker/run_container_blackwell.sh "
            "already does), or have the client send a `mask` file or `click` field.")

    mask, score = click_for_mask(predictor, rgb)
    if mask is None:
        raise RuntimeError("operator aborted the selection")
    return mask, score, "sam2(interactive click)"


# ── Result rendering ──────────────────────────────────────────────────────

def render_overlay(rgb, pose, K):
    """The pose drawn on the frame: oriented box + axes. This is the artifact to
    look at first when a pose comes back wrong -- it usually shows the mask
    grabbed the wrong thing, or the object is symmetric and flipped."""
    center_pose = pose @ np.linalg.inv(TO_ORIGIN)
    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=BBOX)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                        thickness=3, transparency=0, is_input_rgb=True)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def png_b64(image):
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError("failed to PNG-encode an output image")
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "mesh_path": MESH_PATH,
        "mesh_scale": MESH_SCALE,
        "est_refine_iter": EST_REFINE_ITER,
        "interactive": bool(os.environ.get('DISPLAY')),
        "busy": _lock.locked(),
    })


@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    if not _lock.acquire(blocking=False):
        return jsonify({"status": "error",
                        "message": "busy with another request"}), 503
    try:
        try:
            rgb_file = request.files['rgb']
            depth_file = request.files['depth']
            camera_file = request.files['camera']
        except KeyError as missing:
            return jsonify({"status": "error",
                            "message": f"missing required file field: {missing}"}), 400

        rgb_path = os.path.join(DATA_DIR, "rgb.png")
        depth_path = os.path.join(DATA_DIR, "depth.png")
        camera_path = os.path.join(DATA_DIR, "camera.json")
        rgb_file.save(rgb_path)
        depth_file.save(depth_path)
        camera_file.save(camera_path)
        for path in (DATA_DIR, rgb_path, depth_path, camera_path):
            _give_back_ownership(path)

        K, depth_scale = read_camera(camera_path)
        rgb = read_rgb(rgb_path)
        depth = read_depth(depth_path, depth_scale)
        if rgb.shape[:2] != depth.shape[:2]:
            return jsonify({
                "status": "error",
                "message": f"rgb {rgb.shape[:2]} and depth {depth.shape[:2]} differ in size; "
                           "the depth must be registered to the colour frame",
            }), 400

        mask, mask_score, mask_source = resolve_mask(SAM2, rgb)
        if not mask.any():
            return jsonify({"status": "error", "message": "the mask is empty"}), 400
        if not (depth[mask] > 0.001).any():
            return jsonify({
                "status": "error",
                "message": "no valid depth inside the mask -- the object is out of the "
                           f"depth range, or beyond ZFAR={ZFAR}m",
            }), 400

        logging.info(f"registering: mask from {mask_source}, {int(mask.sum())} px")
        started = time.monotonic()
        pose = EST.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=EST_REFINE_ITER)
        elapsed = time.monotonic() - started

        # The winning hypothesis' score from the scoring network. Note this is an
        # unnormalized ranking score (tens, not 0-1) used to sort hypotheses
        # against each other -- it is NOT a probability like SAM-6D's score, so
        # don't carry a SAM-6D `min_score` threshold over unchanged.
        score = float(EST.scores[0]) if getattr(EST, 'scores', None) is not None else 0.0
        vis = render_overlay(rgb, pose, K)
        mask_png = (mask.astype(np.uint8) * 255)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "vis_pose.png"), vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "mask.png"), mask_png)
        np.savetxt(os.path.join(OUTPUT_DIR, "ob_in_cam.txt"), pose.reshape(4, 4))
        for name in ("vis_pose.png", "mask.png", "ob_in_cam.txt"):
            _give_back_ownership(os.path.join(OUTPUT_DIR, name))

        logging.info(f"registered in {elapsed:.2f}s  score={score:.3f}  "
                     f"t={pose[:3, 3].round(4)}m")
        return jsonify({
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "units": "m",
            "pose": pose.reshape(4, 4).tolist(),
            "score": score,
            "mask_score": mask_score,
            "mask_source": mask_source,
            "artifacts": {"mask.png": png_b64(mask_png), "vis_pose.png": png_b64(vis)},
        })
    except Exception as exc:  # noqa: BLE001 - report anything back to the client
        logging.exception("registration failed")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        _lock.release()


set_logging_format()
set_seed(0)

EST, MESH = load_estimator()
TO_ORIGIN, EXTENTS = trimesh.bounds.oriented_bounds(MESH)
BBOX = np.stack([-EXTENTS / 2, EXTENTS / 2], axis=0).reshape(2, 3)
SAM2 = load_sam2()


if __name__ == '__main__':
    logging.info(f"FoundationPose server listening on :{PORT}")
    # threaded=False keeps requests on the main thread, which the OpenCV click
    # window requires (Qt only draws from the main thread). It also serializes
    # the single-GPU pipeline for free; the lock above is belt-and-braces.
    app.run(host='0.0.0.0', port=PORT, threaded=False)
