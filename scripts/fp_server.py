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
- **It does not know which CAD it is looking at, so it works that out first.**
  A PPF classifier (`ppf_classifier.py`) holds a library built from every .ply in
  `CAD_DIR`. Once SAM2 has produced the mask, the masked depth becomes a small
  oriented point cloud, that cloud is matched against every model in the library,
  and the winner *names* the CAD that registration then runs against. The
  classifier returns a name and nothing else -- pose estimation is FoundationPose's
  job, and it starts from scratch on the CAD it is given. Set `PPF_ENABLE=0` to
  skip all of this and pin the server to `MESH_PATH`, which is what it used to do.
- **The model stays resident.** SAM-6D re-ran `demo.sh` (template rendering ->
  segmentation -> pose) as a subprocess on every request. Here the refiner/scorer
  networks and SAM2 are loaded once at startup, and the CAD is swapped in per
  request via `reset_object()` -- the expensive part is the networks, not the mesh,
  so classification does not cost a reload.
- **One object, one pose.** SAM-6D returned a ranked list of detections; this
  returns the single pose of the thing you clicked, as a 4x4 matrix.
- **Metres, not millimetres.** The CAD is scaled to metres on load (BOP-style
  .ply files are mm), so the translation comes back in metres. The client must
  not divide by 1000 anymore.

The SAM-6D artifacts the downstream ICP node reads off disk -- `detection_pem.json`
(init pose) and `detection_ism.npz` (the segmentation mask that crops the scene
cloud) -- are still produced, in SAM-6D's exact format, and still base64'd back to
the client. See `sam6d_style_artifacts()`. That node therefore needs no changes
beyond pointing its `results_dir` at where the new bridge saves them.

Endpoints:

    POST /predict_pose   the whole pipeline: mask -> classify -> register -> pose
    POST /classify       classification only, no registration. Same payload as
                         /predict_pose. This is the endpoint to use during bring-up:
                         it returns the full score table in about a second without
                         spending GPU time on a pose you are only going to look at.
    POST /add_model      upload a .ply, index it into the live library, persist it
    GET  /health         what is loaded, what is in the library

Configuration (all optional, via env vars):

    MESH_PATH        fallback CAD, used when PPF is off or classification fails
                                                      (Data/Input/test_objv2_ear.ply)
    MESH_SCALE       multiplied into the mesh on load (0.001 -- i.e. mm -> m)
    PPF_ENABLE       1 to classify, 0 to always use MESH_PATH        (1)
    CAD_DIR          directory scanned for library .ply files (Data/Input)
    PPF_LIBRARY      the library .npz                    (Data/ppf_library.npz)
    PPF_MIN_MARGIN   top-to-runner-up score gap below which the result is flagged
                     ambiguous. 0 means "always accept the winner". Two thin plates
                     of similar size are genuinely indistinguishable from one view;
                     raise this once you have watched real margins, and set
                     PPF_STRICT=1 to refuse rather than guess.        (0.0)
    PPF_TAU          verification inlier radius in metres, i.e. your depth noise (0.008)
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

import io
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

sys.path.insert(0, os.path.join(CODE_DIR, "scripts"))
from ppf_classifier import (  # noqa: E402
    PPFLibrary,
    PPFParams,
    QueryParams,
    find_ply_files,
    scene_cloud_from_mask,
)

app = Flask(__name__)

DATA_DIR = os.path.join(CODE_DIR, "Data", "Input")
OUTPUT_DIR = os.path.join(CODE_DIR, "Data", "Output", "foundationpose_results")

MESH_PATH = os.environ.get("MESH_PATH", os.path.join(DATA_DIR, "test_objv2_ear.ply"))
MESH_SCALE = float(os.environ.get("MESH_SCALE", "0.001"))
SYMMETRY_INFO = os.environ.get("SYMMETRY_INFO", "")
EST_REFINE_ITER = int(os.environ.get("EST_REFINE_ITER", "5"))
ZFAR = float(os.environ.get("ZFAR", "3.0"))

PPF_ENABLE = os.environ.get("PPF_ENABLE", "1") not in ("0", "false", "False")
CAD_DIR = os.environ.get("CAD_DIR", DATA_DIR)
PPF_LIBRARY = os.environ.get("PPF_LIBRARY", os.path.join(CODE_DIR, "Data", "ppf_library.npz"))
PPF_MIN_MARGIN = float(os.environ.get("PPF_MIN_MARGIN", "0.0"))
PPF_STRICT = os.environ.get("PPF_STRICT", "0") not in ("0", "false", "False")
PPF_TAU = float(os.environ.get("PPF_TAU", "0.008"))

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

_mesh_cache = {}


def load_cad(path):
    """A CAD mesh in metres, plus the oriented-bounds box the overlay is drawn from.

    Cached because the classifier may pick the same part on request after request,
    and `trimesh.bounds.oriented_bounds` is not free on a 250k-vertex scan.
    """
    if path in _mesh_cache:
        return _mesh_cache[path]
    mesh = trimesh.load(path, force='mesh')
    if MESH_SCALE != 1.0:
        mesh.apply_scale(MESH_SCALE)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

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
    logging.info(f"loaded CAD {os.path.basename(path)}: extents={mesh.extents.round(4)}m "
                 f"texture={tex}")
    _mesh_cache[path] = (mesh, to_origin, bbox)
    return _mesh_cache[path]


def load_estimator():
    """One FoundationPose, holding whichever CAD was used last.

    The scorer and refiner networks are the expensive part of construction and they
    are object-agnostic, so swapping CAD between requests is `reset_object()` on
    this one instance rather than a second FoundationPose. That is what makes it
    affordable to let the classifier choose the mesh per request.
    """
    mesh, _, _ = load_cad(MESH_PATH)
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
    est._loaded_cad = MESH_PATH
    logging.info(f"fallback mesh={MESH_PATH} scale={MESH_SCALE} "
                 f"symmetry_tfs={0 if symmetry_tfs is None else len(symmetry_tfs)}")
    return est


def use_cad(path):
    """Point the estimator at a CAD file. Returns (mesh, to_origin, bbox)."""
    mesh, to_origin, bbox = load_cad(path)
    if getattr(EST, '_loaded_cad', None) != path:
        started = time.monotonic()
        EST.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh)
        EST._loaded_cad = path
        logging.info(f"switched CAD -> {os.path.basename(path)} "
                     f"({time.monotonic()-started:.2f}s)")
    return mesh, to_origin, bbox


def load_ppf_library():
    """The CAD library, rebuilt from CAD_DIR if the .npz is missing or stale.

    Auto-building on a missing .npz is deliberate: the alternative is a server that
    starts fine and then fails on the first live trigger because nobody ran the
    offline step. Note that a rebuild only re-derives the sampled clouds -- OpenCV
    cannot serialize a trained PPF3DDetector, so the hashtables are built in-process
    either way, at roughly a second or two per model.
    """
    if not PPF_ENABLE:
        logging.info("PPF_ENABLE=0: classification off, every request uses MESH_PATH")
        return None
    try:
        if os.path.exists(PPF_LIBRARY):
            return PPFLibrary.load(PPF_LIBRARY, train=True)
        paths = find_ply_files(CAD_DIR)
        if not paths:
            logging.warning(f"no .ply files in CAD_DIR={CAD_DIR}; classification off")
            return None
        logging.info(f"no library at {PPF_LIBRARY}; building it from {len(paths)} "
                     f"CAD file(s) in {CAD_DIR}")
        lib = PPFLibrary.build(paths, PPFParams(mesh_scale=MESH_SCALE))
        if not lib.models:
            return None
        lib.save(PPF_LIBRARY)
        _give_back_ownership(PPF_LIBRARY)
        lib.train()
        return lib
    except Exception as exc:  # noqa: BLE001 - a broken library must not stop the server
        logging.exception(f"could not load the PPF library ({exc}); classification off")
        return None


def cad_path_for(name):
    """Library model name -> the .ply on disk that FoundationPose should register."""
    rec = LIBRARY.get(name) if LIBRARY else None
    if rec is None:
        return None
    path = os.path.join(CAD_DIR, rec.filename)
    return path if os.path.exists(path) else None


def classify_object(depth, K, mask):
    """Mask + depth -> the name of the CAD to register against.

    Returns (cad_path, report). `report` is the full score table and always goes
    back to the client, winner or not -- the runner-up margin is the number worth
    watching during bring-up, and it costs nothing to carry.

    A pose *is* computed inside the classifier, for each candidate, as the only way
    to ask "does this CAD explain this cloud". It is used to compute the score and
    then dropped on the floor. Nothing downstream of here sees it; FoundationPose
    below re-estimates the pose from scratch.
    """
    if LIBRARY is None:
        return None, {'ok': False, 'message': 'classification is off'}

    started = time.monotonic()
    pts, nrm = scene_cloud_from_mask(depth, K, mask)
    if len(pts) < 20:
        return None, {'ok': False, 'scores': {},
                      'message': f'only {len(pts)} points survived masking the depth'}

    report = LIBRARY.classify(pts, nrm, K=K, q=QueryParams(
        tau_m=PPF_TAU, min_margin=PPF_MIN_MARGIN))
    report['elapsed_sec'] = round(time.monotonic() - started, 2)

    if not report.get('ok'):
        return None, report
    path = cad_path_for(report['object_name'])
    if path is None:
        report['ok'] = False
        report['message'] = (f"classified as '{report['object_name']}' but "
                             f"{report.get('object_file')} is not in CAD_DIR={CAD_DIR}")
        return None, report

    top = report['scores']
    logging.info(f"PPF: {report['object_name']} score={report['score']:.3f} "
                 f"margin={report['margin']:.3f} over {report.get('runner_up')} "
                 f"({report['scene_points']} pts, {report['elapsed_sec']:.1f}s) "
                 f"table={dict(list(top.items())[:4])}")
    if report.get('ambiguous'):
        logging.warning(
            f"PPF margin {report['margin']:.3f} is below PPF_MIN_MARGIN="
            f"{PPF_MIN_MARGIN}: '{report['object_name']}' and "
            f"'{report.get('runner_up')}' are not separable from this view")
    return path, report


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

def render_overlay(rgb, pose, K, to_origin, bbox):
    """The pose drawn on the frame: oriented box + axes. This is the artifact to
    look at first when a pose comes back wrong -- it usually shows the mask
    grabbed the wrong thing, the object is symmetric and flipped, or (new failure
    mode) the classifier named the wrong CAD, in which case the box is the right
    shape for a part that is not there."""
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                        thickness=3, transparency=0, is_input_rgb=True)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def sam6d_style_artifacts(mask, pose, score, object_name=None):
    """Re-emit the result in SAM-6D's on-disk format.

    The downstream ICP node does not take the mask off the ROS message -- its
    `_load_detection()` reads `detection_pem.json` for the init pose and
    `detection_ism.npz['segmentation']` (K,H,W) for the mask that crops the
    scene cloud. Writing those two files keeps that node working untouched.

    Note the translation here is in MILLIMETRES, because that is the SAM-6D
    convention the ICP node already divides out. The HTTP `pose` field remains
    metres -- these files are a compatibility shim, not the primary contract.

    `obj_name` is an extra key SAM-6D never wrote. It carries the classifier's
    answer to any consumer that reads these files off disk instead of the ROS
    message, so a tracking node can load the right .ply without being rewritten to
    speak HTTP. Readers that do not know the key ignore it.
    """
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    pem = [{
        "scene_id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": bbox,
        "score": score,
        "time": 0.0,
        "R": pose[:3, :3].tolist(),
        "t": (pose[:3, 3] * 1000.0).tolist(),
        "obj_name": object_name,
    }]
    # (K,H,W) with K=1: FoundationPose registers the one object you clicked, so
    # the ICP node's argmax-over-detections picks index 0.
    buf = io.BytesIO()
    np.savez_compressed(buf, segmentation=mask[None].astype(bool))
    return json.dumps(pem).encode('utf-8'), buf.getvalue()


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "fallback_mesh_path": MESH_PATH,
        "mesh_scale": MESH_SCALE,
        "est_refine_iter": EST_REFINE_ITER,
        "interactive": bool(os.environ.get('DISPLAY')),
        "busy": _lock.locked(),
        "ppf": {
            "enabled": LIBRARY is not None,
            "library": PPF_LIBRARY if LIBRARY is not None else None,
            "cad_dir": CAD_DIR,
            "models": LIBRARY.names if LIBRARY is not None else [],
            "min_margin": PPF_MIN_MARGIN,
            "strict": PPF_STRICT,
            "tau_m": PPF_TAU,
        },
    })


def _receive_frame():
    """Save the three uploaded files and decode them. Shared by both endpoints.

    Returns (K, rgb, depth) or raises ValueError with a client-facing message.
    """
    for field in ('rgb', 'depth', 'camera'):
        if field not in request.files:
            raise ValueError(f"missing required file field: '{field}'")

    rgb_path = os.path.join(DATA_DIR, "rgb.png")
    depth_path = os.path.join(DATA_DIR, "depth.png")
    camera_path = os.path.join(DATA_DIR, "camera.json")
    request.files['rgb'].save(rgb_path)
    request.files['depth'].save(depth_path)
    request.files['camera'].save(camera_path)
    for path in (DATA_DIR, rgb_path, depth_path, camera_path):
        _give_back_ownership(path)

    K, depth_scale = read_camera(camera_path)
    rgb = read_rgb(rgb_path)
    depth = read_depth(depth_path, depth_scale)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"rgb {rgb.shape[:2]} and depth {depth.shape[:2]} differ in "
                         "size; the depth must be registered to the colour frame")
    return K, rgb, depth


@app.route('/classify', methods=['POST'])
def classify():
    """Classification only -- mask, then PPF, then stop. No registration.

    Same payload as /predict_pose. This is the endpoint for bring-up: it answers in
    about a second and hands back the entire score table, so you can watch how the
    margin behaves across viewpoints and lighting without spending GPU time on a
    pose you were only ever going to glance at.
    """
    if not _lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "busy with another request"}), 503
    try:
        K, rgb, depth = _receive_frame()
        mask, mask_score, mask_source = resolve_mask(SAM2, rgb)
        if not mask.any():
            return jsonify({"status": "error", "message": "the mask is empty"}), 400

        cad_path, report = classify_object(depth, K, mask)
        return jsonify({
            "status": "success" if report.get('ok') else "error",
            "object_name": report.get('object_name'),
            "object_file": report.get('object_file'),
            "cad_path": cad_path,
            "mask_score": mask_score,
            "mask_source": mask_source,
            "classification": report,
        }), (200 if report.get('ok') else 422)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logging.exception("classification failed")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        _lock.release()


@app.route('/add_model', methods=['POST'])
def add_model():
    """Take a new .ply into the live library, and persist it.

        curl -F model=@new_part.ply http://host:5000/add_model

    The motivating case is a part assembled from two existing models and exported as
    a new mesh: it has to become classifiable without restarting a server that took a
    minute to load its networks. Adding is genuinely incremental -- each model owns
    its own OpenCV detector, so nothing else in the library is retrained.

    Optional form fields:
        name            override the model name (default: the .ply's stem)
        sampling_step   per-model PPF sampling step. Thin curved parts need a finer
                        step (~0.025) than machined blocks; ppf_selftest.py tells you
                        which ones, by scoring each model against itself.
    """
    if LIBRARY is None:
        return jsonify({"status": "error",
                        "message": "classification is off, so there is no library to add to"}), 409
    if not _lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "busy with another request"}), 503
    try:
        if 'model' not in request.files:
            return jsonify({"status": "error",
                            "message": "missing required file field: 'model'"}), 400
        upload = request.files['model']
        name = request.form.get('name') or os.path.splitext(
            os.path.basename(upload.filename or 'model.ply'))[0]
        if not name or os.path.sep in name or name.startswith('.'):
            return jsonify({"status": "error", "message": f"unusable model name {name!r}"}), 400
        step = request.form.get('sampling_step')

        os.makedirs(CAD_DIR, exist_ok=True)
        path = os.path.join(CAD_DIR, f"{name}.ply")
        upload.save(path)
        _give_back_ownership(path)

        rec = LIBRARY.add_model(path, sampling_step=float(step) if step else None)
        LIBRARY.save(PPF_LIBRARY)
        _give_back_ownership(PPF_LIBRARY)
        # A new CAD invalidates nothing about the meshes already loaded, but if this
        # replaced a model of the same name the cached trimesh is now stale.
        _mesh_cache.pop(path, None)

        logging.info(f"added CAD '{rec.name}' to the library ({len(LIBRARY.models)} total)")
        return jsonify({
            "status": "success",
            "object_name": rec.name,
            "object_file": rec.filename,
            "cad_path": path,
            "diameter_m": round(rec.diameter, 5),
            "extents_m": np.asarray(rec.extents).round(5).tolist(),
            "model_points": int(len(rec.cloud)),
            "models": LIBRARY.names,
            # A newly exported part is exactly where a wrong FreeCAD export unit
            # shows up, and the extent pre-filter compares physical size, so a
            # mis-scaled model would be quietly unclassifiable. Say it here.
            "scale_warning": rec.scale_warning,
            # Two parts of near-identical size cannot be told apart by the extent
            # pre-filter, so the new model's arrival may have made an existing one
            # ambiguous. Say so now rather than at the first confused trigger.
            "similar_diameter": [m.name for m in LIBRARY.models
                                 if m.name != rec.name and rec.diameter > 0
                                 and abs(m.diameter - rec.diameter) / rec.diameter < 0.10],
        })
    except Exception as exc:  # noqa: BLE001
        logging.exception("could not add the model")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        _lock.release()


@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    if not _lock.acquire(blocking=False):
        return jsonify({"status": "error",
                        "message": "busy with another request"}), 503
    try:
        try:
            K, rgb, depth = _receive_frame()
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        mask, mask_score, mask_source = resolve_mask(SAM2, rgb)
        if not mask.any():
            return jsonify({"status": "error", "message": "the mask is empty"}), 400
        if not (depth[mask] > 0.001).any():
            return jsonify({
                "status": "error",
                "message": "no valid depth inside the mask -- the object is out of the "
                           f"depth range, or beyond ZFAR={ZFAR}m",
            }), 400

        # --- which CAD? ---------------------------------------------------
        # PPF names the part; that name selects the mesh; FoundationPose then
        # estimates the pose from that mesh, from scratch. The classifier's own
        # internal pose hypotheses do not survive this line.
        cad_path, report = classify_object(depth, K, mask)
        if cad_path is None:
            if PPF_STRICT and LIBRARY is not None:
                return jsonify({
                    "status": "error",
                    "message": f"PPF could not name the object: "
                               f"{report.get('message', 'no model scored above zero')}",
                    "classification": report,
                }), 422
            if LIBRARY is not None:
                logging.warning(f"PPF gave no usable answer "
                                f"({report.get('message', 'no model scored')}); "
                                f"falling back to MESH_PATH={os.path.basename(MESH_PATH)}")
            cad_path = MESH_PATH
        elif report.get('ambiguous') and PPF_STRICT:
            return jsonify({
                "status": "error",
                "message": f"PPF margin {report['margin']:.3f} < PPF_MIN_MARGIN "
                           f"{PPF_MIN_MARGIN}: cannot separate "
                           f"'{report['object_name']}' from '{report.get('runner_up')}'",
                "classification": report,
            }), 422

        mesh, to_origin, bbox = use_cad(cad_path)
        object_name = os.path.splitext(os.path.basename(cad_path))[0]

        logging.info(f"registering {object_name}: mask from {mask_source}, "
                     f"{int(mask.sum())} px")
        started = time.monotonic()
        pose = EST.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=EST_REFINE_ITER)
        elapsed = time.monotonic() - started

        # The winning hypothesis' score from the scoring network. Note this is an
        # unnormalized ranking score (tens, not 0-1) used to sort hypotheses
        # against each other -- it is NOT a probability like SAM-6D's score, so
        # don't carry a SAM-6D `min_score` threshold over unchanged.
        score = float(EST.scores[0]) if getattr(EST, 'scores', None) is not None else 0.0
        vis = render_overlay(rgb, pose, K, to_origin, bbox)
        mask_png = (mask.astype(np.uint8) * 255)
        pem_bytes, ism_bytes = sam6d_style_artifacts(mask, pose, score, object_name)

        # Everything the client needs to reconstruct the result byte-for-byte on
        # its side. detection_pem.json / detection_ism.npz are what the ICP node
        # actually consumes; mask.png / vis_pose.png are for looking at.
        artifacts = {
            "detection_pem.json": pem_bytes,
            "detection_ism.npz": ism_bytes,
            "mask.png": cv2.imencode('.png', mask_png)[1].tobytes(),
            "vis_pose.png": cv2.imencode('.png', vis)[1].tobytes(),
            # The classifier's answer as a plain file, for anything downstream that
            # watches the results directory rather than parsing the reply.
            "object_name.txt": object_name.encode('utf-8'),
        }
        for name, payload in artifacts.items():
            path = os.path.join(OUTPUT_DIR, name)
            with open(path, 'wb') as fh:
                fh.write(payload)
            _give_back_ownership(path)

        logging.info(f"registered {object_name} in {elapsed:.2f}s  score={score:.3f}  "
                     f"t={pose[:3, 3].round(4)}m")
        return jsonify({
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "units": "m",
            "pose": pose.reshape(4, 4).tolist(),
            "score": score,
            # What PPF decided, and how confidently. The bridge republishes
            # object_name so the tracking client knows which .ply to load.
            "object_name": object_name,
            "object_file": os.path.basename(cad_path),
            "classification": report,
            "mask_score": mask_score,
            "mask_source": mask_source,
            "artifacts": {name: base64.b64encode(payload).decode('ascii')
                          for name, payload in artifacts.items()},
        })
    except Exception as exc:  # noqa: BLE001 - report anything back to the client
        logging.exception("registration failed")
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        _lock.release()


set_logging_format()
set_seed(0)

EST = load_estimator()
LIBRARY = load_ppf_library()
SAM2 = load_sam2()


if __name__ == '__main__':
    logging.info(f"FoundationPose server listening on :{PORT}")
    # threaded=False keeps requests on the main thread, which the OpenCV click
    # window requires (Qt only draws from the main thread). It also serializes
    # the single-GPU pipeline for free; the lock above is belt-and-braces.
    app.run(host='0.0.0.0', port=PORT, threaded=False)
