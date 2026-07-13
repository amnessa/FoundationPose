import os
import json
import time
import base64
import threading
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Input")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Output")

# CAD model is fixed per deployed object (client only sends rgb/depth/camera).
# Override with the CAD_PATH env var if needed.
CAD_PATH = os.environ.get("CAD_PATH", os.path.join(DATA_DIR, "test_objv2_base.ply"))

# Instance-segmentation backbone. FastSAM (~2-3 GB VRAM) is the default so the
# pipeline fits on an 8 GB GPU; set SEGMENTOR_MODEL=sam for the heavier, slightly
# more accurate SAM ViT-H (~7-8 GB). demo.sh honours this env var.
SEGMENTOR_MODEL = os.environ.get("SEGMENTOR_MODEL", "fastsam")

# Reduce CUDA fragmentation so peak-VRAM stages don't OOM on small GPUs.
PYTORCH_CUDA_ALLOC_CONF = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

# Raw pipeline artifacts returned to the client (base64-encoded) alongside the
# parsed pose, relative to <OUTPUT_DIR>/sam6d_results. Small (~tens of KB), so
# embedding them in the JSON reply is simpler than a separate download endpoint.
RESULTS_SUBDIR = "sam6d_results"
ARTIFACT_FILES = ("detection_ism.json", "detection_ism.npz", "detection_pem.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In Docker we run as root on a bind-mounted workspace, so files we write are root-owned
# on the host and the host user (uid 1000) can't overwrite them afterwards. run_container.sh
# passes the host identity in; give ownership back for anything we create. demo.sh does the
# same for Data/Output. Unset outside Docker -> no-op.
HOST_UID = os.environ.get("HOST_UID")
HOST_GID = os.environ.get("HOST_GID")


def _give_back_ownership(path):
    if not HOST_UID or os.geteuid() != 0:
        return
    try:
        os.chown(path, int(HOST_UID), int(HOST_GID or HOST_UID))
    except OSError as err:
        print(f"WARNING: could not chown {path} back to the host user: {err}")

# The pipeline is GPU-bound and single-GPU; running two at once OOMs. Serialize
# requests instead of letting them race (the bridge may trigger repeatedly).
_pipeline_lock = threading.Lock()


def _collect_artifacts(results_dir):
    """base64-encode the raw pipeline outputs so the client can write them back
    to disk byte-for-byte (the .npz is binary). Missing files are skipped rather
    than failing the request."""
    artifacts = {}
    for name in ARTIFACT_FILES:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                artifacts[name] = base64.b64encode(fh.read()).decode("ascii")
        else:
            print(f"WARNING: expected artifact missing: {path}")
    return artifacts


def _slim_detection(det):
    """Drop the bulky per-detection RLE mask the pose client never reads.

    The bridge only consumes R / t / score / category_id / bbox; the
    `segmentation` counts array is often larger than the rest of the payload
    combined, so stripping it keeps the HTTP response small.
    """
    return {k: v for k, v in det.items() if k != "segmentation"}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "segmentor_model": SEGMENTOR_MODEL,
        "cad_path": CAD_PATH,
        "busy": _pipeline_lock.locked(),
    })


@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    # Single-flight: reject rather than queue so the client fails fast instead
    # of blocking on a long GPU run it can't see.
    if not _pipeline_lock.acquire(blocking=False):
        return jsonify({
            "status": "error",
            "message": "pipeline busy with another request",
        }), 503

    try:
        # Receive the input data
        try:
            rgb_input = request.files['rgb']
            depth_input = request.files['depth']
            camera_intrinsics = request.files['camera']
        except KeyError as missing:
            return jsonify({
                "status": "error",
                "message": f"missing required file field: {missing}",
            }), 400

        # Overwrite the input files
        rgb_path = os.path.join(DATA_DIR, "rgb.png")
        depth_path = os.path.join(DATA_DIR, "depth.png")
        camera_path = os.path.join(DATA_DIR, "camera.json")

        rgb_input.save(rgb_path)
        depth_input.save(depth_path)
        camera_intrinsics.save(camera_path)
        for path in (DATA_DIR, rgb_path, depth_path, camera_path):
            _give_back_ownership(path)

        print("Input files received and saved.")

        # Trigger the full SAM-6D pipeline via demo.sh
        # (render templates -> instance segmentation -> pose estimation)
        env = os.environ.copy()
        env["CAD_PATH"] = CAD_PATH
        env["RGB_PATH"] = rgb_path
        env["DEPTH_PATH"] = depth_path
        env["CAMERA_PATH"] = camera_path
        env["OUTPUT_DIR"] = OUTPUT_DIR
        env["SEGMENTOR_MODEL"] = SEGMENTOR_MODEL
        env["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

        print(f"Running SAM-6D (segmentor={SEGMENTOR_MODEL})...")
        started = time.monotonic()
        process = subprocess.run(
            ["bash", "demo.sh"],
            cwd=BASE_DIR,
            env=env,
            capture_output=True,
            text=True
        )
        elapsed = time.monotonic() - started

        if process.returncode != 0:
            print("Error during inference:")
            print(process.stderr)
            return jsonify({
                "status": "error",
                "message": "Inference failed",
                "details": process.stderr[-2000:],
            }), 500

        # Pose estimation writes its result here
        results_dir = os.path.join(OUTPUT_DIR, RESULTS_SUBDIR)
        output_file = os.path.join(results_dir, "detection_pem.json")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                prediction = json.load(f)
            poses = [_slim_detection(d) for d in prediction]
            artifacts = _collect_artifacts(results_dir)
            print(f"Inference OK: {len(poses)} detection(s) in {elapsed:.1f}s; "
                  f"returning artifacts: {list(artifacts)}")
            return jsonify({
                "status": "success",
                "elapsed_sec": round(elapsed, 2),
                "pose": poses,
                "artifacts": artifacts,
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Prediction file not found. Inference may have failed.",
                "raw_log": process.stdout[-2000:],
            }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        _pipeline_lock.release()


if __name__ == '__main__':
    # threaded=False keeps the single-GPU pipeline strictly serial; the lock
    # above is the real guard, but this avoids spawning idle worker threads.
    app.run(host='0.0.0.0', port=5000, threaded=True)
