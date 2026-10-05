"""Live tracking for fp_server.py: FoundationPose.track_one behind a WebSocket.

realtime_fp.md S2/S7/S9/S11, the server half. The wire format lives in
`fp_stream.py`; this file is everything that needs the GPU.

- `Tracker` owns the tracking state on top of the server's one `FoundationPose`
  (`EST`). Every touch of `EST` happens under the server's `GPU_LOCK`, so a
  `/predict_pose` registration simply pauses tracking, and its pose becomes the
  tracker's next starting point (`on_registered`).
- `TrackingServer` is a WebSocket thread. Per connection, a receiver loop parses
  frame headers and drops them into a latest-wins slot; a worker thread takes the
  newest frame, tracks it and replies. The client never waits for pose N before
  sending frame N+1, and a slow GPU drops frames instead of queueing them.

Poses: FoundationPose keeps `pose_last` in its *centred-mesh* frame and returns
`pose_last @ tf_to_centered`. Everything on the wire is the CAD frame (the same
convention as /predict_pose), so seeding converts with `inv(tf_to_centered)`.
Ego-motion prediction is a left-multiply (camera side only) and needs no conversion.
"""

import json
import logging
import os
import socket
import threading
import time

import cv2
import numpy as np
import torch

from Utils import nvdiffrast_render
import fp_stream as fs

log = logging.getLogger("fp_tracking")

# Per-session options; a `start` message may override any of them.
DEFAULTS = {
    "refine_iter": 2,          # refiner passes per frame (S6: try 1)
    "fit_tol_m": 0.010,        # a rendered pixel "fits" if measured depth is this close
    "fit_lost": 0.5,           # below this for `lost_frames` frames -> LOST
    "fit_recover": 0.6,        # above this to leave LOST (hysteresis)
    "lost_frames": 3,
    "ego_motion": True,        # predict with T_base_cam before tracking (S7)
    "auto_reseed": True,       # LOST -> register() on a mask projected from the last good pose
    "reseed_interval_s": 1.0,  # register is ~1 s of GPU; do not hammer it
    "reseed_dilate": 0.25,     # mask dilation, as a fraction of the crop radius
}

_THREAD_PREFIX = "fp-track"


def register_cropped(est, K, rgb, depth, mask, iteration=5, margin_px=40):
    """`est.register` on the part of the frame it actually looks at (see
    fp_stream.register_roi). Same pose, a fraction of the GPU memory: a full
    1280x720 register does not fit the 8 GB RTX 4060. Returns the pose (camera frame,
    CAD frame, metres) and sets est.pose_last like register() always does."""
    roi = fs.register_roi(mask, depth, K, est.diameter, margin_px=margin_px)
    if roi is None:
        return est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)
    return est.register(K=fs.shift_K(K, roi), rgb=np.ascontiguousarray(fs.crop(rgb, roi)),
                        depth=np.ascontiguousarray(fs.crop(depth, roi)),
                        ob_mask=np.ascontiguousarray(fs.crop(mask, roi)),
                        iteration=iteration)


class _QuietTrackThreads(logging.Filter):
    """estimater.py and the refiner log several INFO lines per track_one call. At
    30 Hz that buries everything else, so drop the root logger's INFO records from
    the tracking threads. Our own logger ('fp_tracking') and warnings still pass."""

    def filter(self, record):
        return not (record.name == "root" and record.levelno < logging.WARNING
                    and record.threadName.startswith(_THREAD_PREFIX))


def _install_quiet_filter():
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, _QuietTrackThreads) for f in handler.filters):
            handler.addFilter(_QuietTrackThreads())


class Tracker:
    """Tracking state machine: IDLE -> TRACKING <-> LOST.

    `use_cad(name)` switches EST to the named CAD (the server's own helper);
    `register_iter`, `register_crop` and `zfar` mirror the server's registration
    settings (the auto re-seed registers the same way /predict_pose does).
    """

    def __init__(self, est, gpu_lock, use_cad=None, register_iter=5, register_crop=False,
                 zfar=3.0):
        self.est = est
        self.register_crop = register_crop
        self.gpu_lock = gpu_lock
        self.use_cad = use_cad
        self.register_iter = register_iter
        self.zfar = zfar
        self.opts = dict(DEFAULTS)
        self.state = "IDLE"
        self._low = 0              # consecutive frames below fit_lost
        self._good = None          # last pose with a good fit, centred frame (np 4x4)
        self._good_T = None        # T_base_cam of the frame `_good` was measured in
        self._T_prev = None        # T_base_cam of the frame pose_last belongs to
        self._last_reseed = 0.0
        torch.backends.cudnn.benchmark = True
        _install_quiet_filter()

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def object_name(self):
        path = getattr(self.est, "_loaded_cad", None)
        return os.path.splitext(os.path.basename(path))[0] if path else None

    def _tf_c(self):
        return self.est.get_tf_to_centered_mesh().cpu().numpy().astype(np.float64)

    def _set_pose_last(self, pose_c):
        self.est.pose_last = torch.as_tensor(np.asarray(pose_c), dtype=torch.float,
                                             device="cuda").reshape(4, 4)

    def _pose_last(self):
        return self.est.pose_last.reshape(4, 4).cpu().numpy().astype(np.float64)

    def _to_cad(self, pose_c):
        return pose_c @ self._tf_c()

    def _hint(self, pose_c, K_full):
        return fs.projection_hint(pose_c, K_full, self.est.diameter)

    def _render_depth(self, pose_c, K, hw):
        ob = torch.as_tensor(pose_c, dtype=torch.float, device="cuda").reshape(1, 4, 4)
        _, depth, _ = nvdiffrast_render(K=K, H=hw[0], W=hw[1], ob_in_cams=ob,
                                        glctx=self.est.glctx,
                                        mesh_tensors=self.est.mesh_tensors,
                                        output_size=hw)
        return depth[0]

    def _fit(self, pose_c, depth_t, K):
        """S9: fraction of rendered CAD pixels whose measured depth is within
        fit_tol_m. Returns (fit, coverage); coverage = rendered pixels that have any
        measured depth at all, which separates "occluded/wrong" from "no depth"."""
        rendered_d = self._render_depth(pose_c, K, tuple(depth_t.shape))
        rendered = rendered_d > 0
        n = int(rendered.sum())
        if n < 20:
            return 0.0, 0.0
        valid = rendered & (depth_t > 0.001)
        hits = valid & ((depth_t - rendered_d).abs() < self.opts["fit_tol_m"])
        return float(hits.sum()) / n, float(valid.sum()) / n

    # ── control ───────────────────────────────────────────────────────────

    def start(self, msg):
        """`start` message -> reply dict. Raises ValueError on a bad request."""
        opts = dict(DEFAULTS)
        for key in DEFAULTS:
            if key in msg:
                opts[key] = type(DEFAULTS[key])(msg[key])
        seed = msg.get("seed", "last")
        with self.gpu_lock:
            if seed == "pose":
                pose = np.asarray(msg.get("pose"), dtype=np.float64)
                if pose.size != 16:
                    raise ValueError("seed 'pose' needs 'pose': 16 floats (T_cam_obj, metres)")
                if msg.get("object"):
                    if self.use_cad is None:
                        raise ValueError("this server cannot switch CAD")
                    self.use_cad(msg["object"])
                pose_c = pose.reshape(4, 4) @ np.linalg.inv(self._tf_c())
            elif seed == "last":
                if self.est.pose_last is None:
                    raise ValueError("nothing registered yet: call /predict_pose first, "
                                     "or seed with a pose")
                pose_c = self._pose_last()
            else:
                raise ValueError(f"unknown seed {seed!r} (use 'last' or 'pose')")
            self._set_pose_last(pose_c)
            self.opts = opts
            self.state = "TRACKING"
            self._low = 0
            self._good = pose_c
            T = msg.get("T_base_cam")
            self._T_prev = self._good_T = None if T is None else np.reshape(T, (4, 4))
            log.info(f"tracking {self.object_name} (seed={seed}, "
                     f"refine_iter={opts['refine_iter']}, ego_motion={opts['ego_motion']})")
            return {"type": "started", "state": self.state, "object_name": self.object_name,
                    "diameter_m": float(self.est.diameter), "seed": seed}

    def stop(self):
        with self.gpu_lock:
            if self.state != "IDLE":
                log.info("tracking stopped")
            self.state = "IDLE"
            self._good = self._T_prev = self._good_T = None

    def on_registered(self, T_base_cam=None):
        """Called by /predict_pose (inside GPU_LOCK) after a registration: the new
        pose becomes the starting point, CAD switch included (S11)."""
        if self.state == "IDLE":
            return
        self._good = self._pose_last()
        self._T_prev = self._good_T = T_base_cam
        self._low = 0
        self.state = "TRACKING"
        log.info(f"re-seeded from /predict_pose: {self.object_name}")

    # ── per frame ─────────────────────────────────────────────────────────

    def step(self, header, rgb, depth_raw):
        """Track one decoded frame. Returns the reply dict (minus transport fields)."""
        timings = {}
        K_full = np.asarray(header["K"], dtype=np.float64).reshape(3, 3)
        K = fs.shift_K(K_full, header.get("roi"))
        T_now = header.get("T_base_cam")
        T_now = None if T_now is None else np.asarray(T_now, dtype=np.float64).reshape(4, 4)

        t = time.perf_counter()
        depth = depth_raw.astype(np.float32) * float(header["depth"]["scale"])
        depth[(depth < 0.001) | (depth >= self.zfar) | ~np.isfinite(depth)] = 0
        timings["depth_to_m_ms"] = (time.perf_counter() - t) * 1e3

        reply = {"state": self.state, "pose": None, "fit": None, "coverage": None,
                 "hint": None, "reseeded": False}
        with self.gpu_lock:
            t_lock = time.perf_counter()
            timings["gpu_lock_wait_ms"] = (t_lock - t) * 1e3 - timings["depth_to_m_ms"]
            if self.state == "IDLE" or self.est.pose_last is None:
                self.state = reply["state"] = "IDLE"
                return reply, timings
            reply["object_name"] = self.object_name
            depth_t = torch.as_tensor(depth, device="cuda")

            if self.state == "TRACKING":
                pose_c = self._pose_last()
                if self.opts["ego_motion"]:
                    pose_c = fs.predict_ego(pose_c, self._T_prev, T_now)
                    self._set_pose_last(pose_c)
                t = time.perf_counter()
                self.est.track_one(rgb=rgb, depth=depth, K=K,
                                   iteration=self.opts["refine_iter"])
                pose_c = self._pose_last()
                timings["track_ms"] = (time.perf_counter() - t) * 1e3
                t = time.perf_counter()
                fit, cov = self._fit(pose_c, depth_t, K)
                timings["fit_ms"] = (time.perf_counter() - t) * 1e3
                self._T_prev = T_now
                if fit >= self.opts["fit_lost"]:
                    self._low = 0
                    self._good, self._good_T = pose_c, T_now
                else:
                    self._low += 1
                    if self._low >= self.opts["lost_frames"]:
                        self.state = "LOST"
                        log.warning(f"LOST {self.object_name}: fit {fit:.2f} < "
                                    f"{self.opts['fit_lost']} for {self._low} frames")
            else:
                pose_c, fit, cov = self._recover(rgb, depth, depth_t, K, T_now, timings, reply)

            reply["state"] = self.state
            reply["fit"], reply["coverage"] = round(fit, 4), round(cov, 4)
            if self.state == "TRACKING":
                reply["pose"] = self._to_cad(pose_c).tolist()
                # No hint while LOST: the client then sends the full frame.
                reply["hint"] = self._hint(pose_c, K_full)
        return reply, timings

    def _recover(self, rgb, depth, depth_t, K, T_now, timings, reply):
        """LOST: first a cheap track_one from the last good pose (the part may just
        have been briefly covered); then, rate-limited, a full register() on a mask
        projected from that pose (S9). Leaves LOST only above fit_recover."""
        good = self._good
        if self.opts["ego_motion"]:
            good = fs.predict_ego(good, self._good_T, T_now)
        self._set_pose_last(good)
        t = time.perf_counter()
        self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=self.opts["refine_iter"])
        pose_c = self._pose_last()
        fit, cov = self._fit(pose_c, depth_t, K)
        timings["track_ms"] = (time.perf_counter() - t) * 1e3

        if fit < self.opts["fit_recover"] and self.opts["auto_reseed"] and \
                time.monotonic() - self._last_reseed >= self.opts["reseed_interval_s"]:
            self._last_reseed = time.monotonic()
            t = time.perf_counter()
            mask = self._projected_mask(good, K, depth.shape)
            if mask is not None and int(((depth > 0.001) & mask).sum()) >= 50:
                if self.register_crop:
                    register_cropped(self.est, K, rgb, depth, mask,
                                     iteration=self.register_iter)
                else:
                    self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                      iteration=self.register_iter)
                pose_r = self._pose_last()
                fit_r, cov_r = self._fit(pose_r, depth_t, K)
                timings["reseed_ms"] = (time.perf_counter() - t) * 1e3
                log.info(f"re-seed attempt: fit {fit_r:.2f}")
                if fit_r > fit:
                    pose_c, fit, cov = pose_r, fit_r, cov_r
                    reply["reseeded"] = True

        self._T_prev = T_now
        if fit >= self.opts["fit_recover"]:
            self.state = "TRACKING"
            self._low = 0
            self._good, self._good_T = pose_c, T_now
            self._set_pose_last(pose_c)
            log.info(f"recovered {self.object_name}: fit {fit:.2f}"
                     f"{' (re-seeded)' if reply['reseeded'] else ''}")
        else:
            # Stay anchored to the last good pose, not to whatever the failed
            # attempts converged on.
            self._set_pose_last(good)
        return pose_c, fit, cov

    def _projected_mask(self, pose_c, K, hw):
        rendered = (self._render_depth(pose_c, K, hw) > 0).cpu().numpy()
        if not rendered.any():
            return None
        z = max(pose_c[2, 3], 1e-3)
        r_px = K[0, 0] * self.est.diameter * 1.2 / 2.0 / z
        k = max(3, int(self.opts["reseed_dilate"] * r_px) | 1)
        return cv2.dilate(rendered.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


class TrackingServer:
    """WebSocket endpoint, one tracking client at a time (S10)."""

    def __init__(self, tracker, host="0.0.0.0", port=fs.DEFAULT_PORT):
        self.tracker = tracker
        self.host, self.port = host, port
        self._busy = threading.Lock()
        self.stats = {"clients": 0, "frames": 0, "dropped": 0}

    def start(self):
        try:
            from websockets.sync.server import serve  # noqa: F401
        except ImportError:
            log.warning("python package 'websockets' missing: tracking endpoint off "
                        "(pip install websockets)")
            return False
        threading.Thread(target=self._run, name=f"{_THREAD_PREFIX}-server",
                         daemon=True).start()
        return True

    def _run(self):
        from websockets.sync.server import serve
        # compression=None: JPEG/PNG payloads do not deflate, it would only cost CPU.
        with serve(self._handle, self.host, self.port, max_size=64 * 2**20,
                   compression=None) as server:
            log.info(f"tracking endpoint listening on ws://{self.host}:{self.port}")
            server.serve_forever()

    def _handle(self, ws):
        from websockets.exceptions import ConnectionClosed
        peer = ws.remote_address
        send_lock = threading.Lock()

        def send(obj):
            with send_lock:
                ws.send(json.dumps(obj))

        if not self._busy.acquire(blocking=False):
            send({"type": "error", "message": "another client is already tracking"})
            ws.close()
            return
        try:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.stats["clients"] += 1
        log.info(f"tracking client connected: {peer}")
        slot = fs.LatestSlot()
        worker = threading.Thread(target=self._work, args=(slot, send),
                                  name=f"{_THREAD_PREFIX}-worker", daemon=True)
        worker.start()
        try:
            for msg in ws:
                t_recv = time.time()
                try:
                    if isinstance(msg, str):
                        send(self._control(json.loads(msg)))
                    else:
                        header, rgb_b, depth_b = fs.split_frame(msg)
                        slot.put((t_recv, header, rgb_b, depth_b))
                except (ValueError, KeyError, TypeError) as exc:
                    send({"type": "error", "message": str(exc)})
        except ConnectionClosed:
            pass
        finally:
            slot.close()
            worker.join(timeout=10)
            self.tracker.stop()
            self._busy.release()
            log.info(f"tracking client disconnected: {peer}")

    def _control(self, msg):
        kind = msg.get("type")
        if kind == "start":
            return self.tracker.start(msg)
        if kind == "stop":
            self.tracker.stop()
            return {"type": "stopped"}
        raise ValueError(f"unknown control message type {kind!r}")

    def _work(self, slot, send):
        window = {"t0": time.monotonic(), "n": 0, "gpu": 0.0, "lat": 0.0}
        while True:
            item, dropped = slot.get(timeout=0.5)
            if item is None:
                if slot.closed:
                    return
                continue
            t_recv, header, rgb_b, depth_b = item
            t_start = time.time()
            timings = {"queue_ms": (t_start - t_recv) * 1e3}
            try:
                t = time.perf_counter()
                rgb, depth = fs.decode_frame(header, rgb_b, depth_b)
                timings["decode_ms"] = (time.perf_counter() - t) * 1e3
                reply, step_t = self.tracker.step(header, rgb, depth)
                timings.update(step_t)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the session
                log.exception("tracking step failed")
                reply = {"state": self.tracker.state, "pose": None, "error": str(exc)}
            timings["server_ms"] = (time.time() - t_recv) * 1e3
            reply.update({"type": "pose", "seq": header.get("seq"),
                          "stamp": header.get("stamp"), "t_sent": header.get("t_sent"),
                          "dropped": dropped,
                          "timings": {k: round(v, 2) for k, v in timings.items()}})
            try:
                send(reply)
            except Exception:  # noqa: BLE001 - the socket is gone; receiver cleans up
                return
            self.stats["frames"] += 1
            self.stats["dropped"] += dropped
            self._log_rate(window, timings)

    def _log_rate(self, w, timings):
        w["n"] += 1
        w["gpu"] += timings.get("track_ms", 0.0) + timings.get("fit_ms", 0.0)
        w["lat"] += timings["server_ms"]
        elapsed = time.monotonic() - w["t0"]
        if elapsed >= 5.0:
            log.info(f"tracking {self.tracker.state} {self.tracker.object_name}: "
                     f"{w['n'] / elapsed:.1f} Hz, gpu {w['gpu'] / w['n']:.1f} ms, "
                     f"server {w['lat'] / w['n']:.1f} ms/frame")
            w.update(t0=time.monotonic(), n=0, gpu=0.0, lat=0.0)

    def health(self):
        return {"port": self.port, "state": self.tracker.state,
                "object_name": self.tracker.object_name,
                "client_connected": self._busy.locked(), **self.stats}
