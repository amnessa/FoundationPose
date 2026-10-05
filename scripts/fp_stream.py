"""Wire protocol for live FoundationPose tracking (realtime_fp.md, S2-S5, S7).

Pure numpy/OpenCV, no torch: the tracking server (`fp_tracking.py`, in the GPU
container) and every client (`track_replay.py` on any host, the ROS tracker node in
welding_cell_ws) import this same file, so both ends always agree on the format.

One WebSocket connection carries everything:

  client -> server, text (JSON control):
      {"type": "start", "seed": "last"}                  continue from the last /predict_pose
      {"type": "start", "seed": "pose", "pose": [16],    T_cam_obj in metres, CAD frame
       "object": "test_objv2_ear"}                       (optional: switch CAD first)
          optional in either: "refine_iter", "fit_tol_m", "fit_lost", "fit_recover",
          "lost_frames", "ego_motion", "auto_reseed", "reseed_interval_s"
      {"type": "stop"}

  client -> server, binary (one camera frame):
      MAGIC | uint32 header length | header JSON | rgb bytes | depth bytes
      header = {
        "seq": int, "stamp": float (image stamp, s),  "t_sent": float (client clock, echoed),
        "K": [9] FULL-frame intrinsics,  "full_hw": [H, W],
        "roi": [x0, y0, w, h] or null (= full frame; the images are this crop),
        "rgb":   {"enc": "jpeg" | "raw", "len": n},       raw = h*w*3 uint8, RGB order
        "depth": {"enc": "png"  | "raw", "len": n,        raw = h*w uint16 little endian
                  "scale": <raw units -> metres>},          e.g. 0.001 for 16UC1 in mm
        "T_base_cam": [16] or null                        TF at the image stamp (S7)
      }

  server -> client, text (JSON):
      {"type": "started", "object_name", "diameter_m", "hint", "state"}
      {"type": "stopped"} | {"type": "error", "message"}
      {"type": "pose", "seq", "stamp", "t_sent", "state": "TRACKING" | "LOST" | "IDLE",
       "pose": 4x4 T_cam_obj (metres, CAD frame -- same convention as /predict_pose) or null,
       "fit", "coverage", "object_name",
       "hint": {"u", "v", "r"}  where to crop the NEXT frame, full-frame pixels,
       "dropped": frames overwritten in the latest-wins slot since the last reply,
       "reseeded": bool, "timings": {stage: ms}}

Images are always row-major and match the ROI (or the full frame). Poses are always
in the camera frame of the frame they answer; cropping only shifts the principal
point, never the camera frame.
"""

import json
import struct
import threading
import time

import cv2
import numpy as np

MAGIC = b"FPT1"
_HDR = struct.Struct("<4sI")

DEFAULT_PORT = 5001


# ── frames ────────────────────────────────────────────────────────────────

def encode_rgb(rgb, enc="jpeg", quality=90):
    """RGB uint8 (h,w,3) -> bytes."""
    if enc == "raw":
        return np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    if enc == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return buf.tobytes()
    raise ValueError(f"unknown rgb encoding {enc!r}")


def decode_rgb(data, enc, hw):
    if enc == "raw":
        return np.frombuffer(data, np.uint8).reshape(hw[0], hw[1], 3)
    if enc == "jpeg":
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode the JPEG rgb")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"unknown rgb encoding {enc!r}")


def encode_depth(depth, enc="png"):
    """uint16 (h,w) raw sensor units -> bytes. Lossless only: never JPEG depth."""
    depth = np.ascontiguousarray(depth, dtype=np.uint16)
    if enc == "raw":
        return depth.astype("<u2", copy=False).tobytes()
    if enc == "png":
        ok, buf = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise ValueError("PNG encoding failed")
        return buf.tobytes()
    raise ValueError(f"unknown depth encoding {enc!r}")


def decode_depth(data, enc, hw):
    if enc == "raw":
        return np.frombuffer(data, "<u2").reshape(hw[0], hw[1])
    if enc == "png":
        depth = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise ValueError("could not decode the 16-bit PNG depth")
        return depth
    raise ValueError(f"unknown depth encoding {enc!r}")


def pack_frame(seq, stamp, K, full_hw, rgb, depth, depth_scale, roi=None,
               T_base_cam=None, rgb_enc="jpeg", depth_enc="png", jpeg_quality=90):
    """One camera frame -> one binary WebSocket message.

    `rgb`/`depth` are already cropped to `roi` ([x0, y0, w, h]) when one is given;
    `K` is always the full-frame matrix (the server shifts it).
    """
    rgb_b = encode_rgb(rgb, rgb_enc, jpeg_quality)
    depth_b = encode_depth(depth, depth_enc)
    header = {
        "seq": int(seq),
        "stamp": float(stamp),
        "t_sent": time.time(),
        "K": np.asarray(K, dtype=float).reshape(-1).tolist(),
        "full_hw": [int(full_hw[0]), int(full_hw[1])],
        "roi": None if roi is None else [int(v) for v in roi],
        "rgb": {"enc": rgb_enc, "len": len(rgb_b)},
        "depth": {"enc": depth_enc, "len": len(depth_b), "scale": float(depth_scale)},
        "T_base_cam": None if T_base_cam is None
        else np.asarray(T_base_cam, dtype=float).reshape(-1).tolist(),
    }
    hdr = json.dumps(header, separators=(",", ":")).encode()
    return b"".join([_HDR.pack(MAGIC, len(hdr)), hdr, rgb_b, depth_b])


def unpack_frame(msg):
    """Binary message -> (header, rgb uint8 RGB (h,w,3), depth uint16 (h,w))."""
    header, rgb_b, depth_b = split_frame(msg)
    return header, *decode_frame(header, rgb_b, depth_b)


def split_frame(msg):
    """Parse only the header, leave the image bytes encoded (cheap; for the
    receiver thread, which must never fall behind the socket)."""
    if len(msg) < _HDR.size:
        raise ValueError("frame message too short")
    magic, n = _HDR.unpack_from(msg, 0)
    if magic != MAGIC:
        raise ValueError(f"bad frame magic {magic!r}")
    start = _HDR.size + n
    header = json.loads(bytes(msg[_HDR.size:start]))
    n_rgb, n_depth = header["rgb"]["len"], header["depth"]["len"]
    if len(msg) != start + n_rgb + n_depth:
        raise ValueError(f"frame length {len(msg)} != header's {start + n_rgb + n_depth}")
    return header, msg[start:start + n_rgb], msg[start + n_rgb:]


def decode_frame(header, rgb_b, depth_b):
    hw = frame_hw(header)
    rgb = decode_rgb(rgb_b, header["rgb"]["enc"], hw)
    depth = decode_depth(depth_b, header["depth"]["enc"], hw)
    if rgb.shape[:2] != hw or depth.shape != hw:
        raise ValueError(f"image sizes rgb {rgb.shape[:2]} depth {depth.shape} "
                         f"do not match the header's {hw}")
    return rgb, depth


def frame_hw(header):
    roi = header.get("roi")
    if roi is None:
        return tuple(header["full_hw"])
    return (roi[3], roi[2])


# ── ROI (S4) ──────────────────────────────────────────────────────────────

def shift_K(K, roi):
    """Intrinsics of a crop: the principal point moves, nothing else does."""
    K = np.array(K, dtype=np.float64).reshape(3, 3)
    if roi is not None:
        K[0, 2] -= roi[0]
        K[1, 2] -= roi[1]
    return K


def roi_from_hint(hint, full_hw, margin_px=40, min_size=64):
    """Server hint {u, v, r} (+ a motion margin) -> [x0, y0, w, h] clamped to the
    frame, or None when the hint is missing or the square would leave nothing."""
    if not hint:
        return None
    H, W = full_hw
    r = float(hint["r"]) + float(margin_px)
    r = max(r, min_size / 2.0)
    x0 = int(np.floor(hint["u"] - r))
    y0 = int(np.floor(hint["v"] - r))
    x1 = int(np.ceil(hint["u"] + r))
    y1 = int(np.ceil(hint["v"] + r))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def crop(img, roi):
    if roi is None:
        return img
    x0, y0, w, h = roi
    return img[y0:y0 + h, x0:x0 + w]


def register_roi(mask, depth_m, K, diameter, crop_ratio=1.2, margin_px=40):
    """The part of the frame `register()` actually needs, as [x0, y0, w, h].

    Every hypothesis is centred on the mask and rendered/cropped in a square of
    radius f * diameter * crop_ratio / 2 / z around its projection, so nothing
    outside that square (plus the mask's own extent) reaches the networks. Cropping
    to it changes no input to them, but it shrinks the full-frame warps the scorer
    does for all ~250 hypotheses at once, which at 1280x720 need 2.6 GB on their
    own and do not fit an 8 GB GPU. Returns None when the mask has no valid depth.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    z = depth_m[ys, xs]
    z = z[z > 0.001]
    if len(z) == 0:
        return None
    H, W = mask.shape[:2]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    r = K[0, 0] * diameter * crop_ratio / 2.0 / float(np.min(z))
    u, v = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    r = max(r, (xs.max() - xs.min()) / 2.0, (ys.max() - ys.min()) / 2.0) + margin_px
    x0, y0 = max(int(u - r), 0), max(int(v - r), 0)
    x1, y1 = min(int(np.ceil(u + r)), W), min(int(np.ceil(v + r)), H)
    return [x0, y0, x1 - x0, y1 - y0]


def projection_hint(T_cam_center, K, diameter, crop_ratio=1.2):
    """Where FoundationPose will look next: the projected object centre and the
    radius of its crop square, f * (diameter * crop_ratio / 2) / z, in pixels."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    x, y, z = np.asarray(T_cam_center, dtype=np.float64)[:3, 3]
    if z <= 1e-3:
        return None
    return {"u": float(K[0, 0] * x / z + K[0, 2]),
            "v": float(K[1, 1] * y / z + K[1, 2]),
            "r": float(K[0, 0] * diameter * crop_ratio / 2.0 / z)}


# ── ego-motion (S7) ───────────────────────────────────────────────────────

def predict_ego(pose_prev, T_base_cam_prev, T_base_cam_now):
    """Object pose in the new camera, assuming the object stayed put in base:
    inv(T_base_cam_now) @ T_base_cam_prev @ pose_prev. Works for any object-side
    frame (CAD or centred mesh), since only the camera side changes."""
    if T_base_cam_prev is None or T_base_cam_now is None:
        return pose_prev
    T_prev = np.asarray(T_base_cam_prev, dtype=np.float64).reshape(4, 4)
    T_now = np.asarray(T_base_cam_now, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(T_now) @ T_prev @ np.asarray(pose_prev, dtype=np.float64)


# ── latest-wins slot (S2) ─────────────────────────────────────────────────

class LatestSlot:
    """A one-element mailbox: `put` overwrites whatever is waiting (and counts it
    as dropped), `get` blocks for the newest item. The consumer therefore always
    works on the freshest frame and a slow consumer never builds a backlog."""

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._closed = False
        self.dropped = 0

    def put(self, item):
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify()

    def get(self, timeout=None):
        """Newest item, or None on timeout / close. Returns (item, dropped_since_last)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._item is not None or self._closed,
                                       timeout):
                return None, 0
            item, self._item = self._item, None
            dropped, self.dropped = self.dropped, 0
            return item, dropped

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self):
        return self._closed
