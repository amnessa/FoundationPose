#!/usr/bin/env python3
"""Live FoundationPose tracking: camera frames -> local tracker -> ICP-style pose.

The laptop half of the split in realtime_fp.md: registration stays on the desktop
(foundationpose_bridge_node -> fp_server.py /predict_pose, one frame per part), and
this node tracks every frame on the laptop's own GPU through fp_track_server.py,
which runs in the laptop's FoundationPose container and listens on localhost.

    camera ──► this node ── ws://127.0.0.1:5001 ──► fp_track_server.py (container)
      ▲          │  ▲         frames + T_base_cam         track_one, fit, LOST
      │          │  └──────── pose per frame ◄────────────
      │          ├─► /perception/fp/pose (PoseStamped, camera frame), TF camera -> fp_object
      │          └─► on LOST: one frame + projected mask ──► desktop /predict_pose
      │                                                      (no click), restart from it
    /perception/detections (latched, from foundationpose_bridge_node) = the seed

Outputs match the ICP node's: a geometry_msgs/PoseStamped in the camera optical
frame, and TF `camera_color_optical_frame -> <child_frame>`, both stamped with the
*image* stamp, so RViz places the part where the moving camera was. A latched mesh
Marker in <child_frame> (frame_locked) redraws the CAD at every TF update.

Topics
    in   <rgb_topic> <depth_topic> <camera_info_topic>   the camera (paired by stamp)
         <detections_topic>  vision_msgs/Detection3DArray, the bridge's registration
    out  <pose_topic>        geometry_msgs/PoseStamped, every tracked frame
         <status_topic>      std_msgs/String JSON at 2 Hz: state, rate, delay, fit
         <marker_topic>      visualization_msgs/Marker, latched mesh in <child_frame>
         TF                  <camera frame> -> <child_frame>
Services (std_srvs/Trigger)
    ~/start    track from the last detection received
    ~/stop     stop tracking
    ~/reseed   re-register now through the desktop (mask projected from the last pose)

Robot motion (eye-in-hand) is cancelled with T_base_cam looked up from TF at each
image stamp (and at the seed's stamp). Without TF the node still works; the tracker
then sees robot motion as part motion.

Needs `fp_stream.py` (installed as admittance_control/fp_stream.py, or next to this
file) and the `websocket-client` package (python3-websocket).

    ros2 run admittance_control fp_tracker_node.py --ros-args \\
        -p register_url:=http://<desktop>:5000/predict_pose
"""

import base64
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker

try:
    from admittance_control import fp_stream as fs
    from admittance_control.geometry import rotmat_to_quat
except ImportError:  # run from the FoundationPose repo, next to fp_stream.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fp_stream as fs  # noqa: E402

    def rotmat_to_quat(R):
        """3x3 rotation -> [x, y, z, w] (largest-diagonal branch, stable)."""
        R = np.asarray(R, dtype=float)
        t = np.trace(R)
        if t > 0.0:
            s = 0.5 / np.sqrt(t + 1.0)
            return [(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                    (R[1, 0] - R[0, 1]) * s, 0.25 / s]
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        return q

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("fp_tracker_node needs websocket-client: sudo apt install python3-websocket")


# ── small pure helpers ────────────────────────────────────────────────────

def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_to_rotmat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_msg_to_matrix(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat(p.orientation.x, p.orientation.y, p.orientation.z,
                               p.orientation.w)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def transform_to_matrix(t) -> np.ndarray:
    T = np.eye(4)
    r = t.transform.rotation
    T[:3, :3] = quat_to_rotmat(r.x, r.y, r.z, r.w)
    v = t.transform.translation
    T[:3, 3] = [v.x, v.y, v.z]
    return T


def color_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (rgb8/bgr8/rgba8/bgra8) -> contiguous RGB uint8 (h, w, 3)."""
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported colour encoding {msg.encoding!r}')
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
    if msg.encoding.startswith('bgr'):
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def depth_to_mm(msg: Image) -> np.ndarray:
    """16UC1/mono16 (mm) or 32FC1 (m) -> uint16 mm (h, w); invalid = 0."""
    if msg.encoding in ('16UC1', 'mono16'):
        dt = np.dtype('>u2' if msg.is_bigendian else '<u2')
        rows = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)
        return np.ascontiguousarray(rows[:, :msg.width], dtype=np.uint16)
    if msg.encoding == '32FC1':
        dt = np.dtype('>f4' if msg.is_bigendian else '<f4')
        rows = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 4)
        m = rows[:, :msg.width]
        mm = np.where(np.isfinite(m) & (m > 0), np.round(m * 1000.0), 0)
        return np.clip(mm, 0, 65535).astype(np.uint16)
    raise ValueError(f'unsupported depth encoding {msg.encoding!r}')


class FpTrackerNode(Node):

    def __init__(self) -> None:
        super().__init__('fp_tracker')
        p = self.declare_parameter
        p('track_url', 'ws://127.0.0.1:5001')
        p('register_url', 'http://127.0.0.1:5000/predict_pose')
        p('rgb_topic', '/camera/color/image_raw')
        p('depth_topic', '/camera/depth/image_rect_raw')
        p('camera_info_topic', '/camera/color/camera_info')
        p('detections_topic', '/perception/detections')
        p('pose_topic', '/perception/fp/pose')
        p('status_topic', '/perception/fp/status')
        p('marker_topic', '/perception/fp/marker')
        p('base_frame', 'base_link')
        p('child_frame', 'fp_object')
        # Seed automatically from every new detection the bridge publishes. A latched
        # detection older than this (e.g. from before this node started) is ignored;
        # call ~/start to use it anyway.
        p('auto_start', True)
        p('max_seed_age_sec', 30.0)
        p('max_sync_delta_sec', 0.005)   # RGB/depth pairing; the camera node shares stamps
        p('max_in_flight', 2)
        p('refine_iter', 2)
        # The tracker is on localhost, so raw bytes: encoding JPEG/PNG costs ~20-30 ms
        # of delay for nothing (measured ~100 ms -> ~75 ms). jpeg/png only if the
        # tracker ever moves to another machine.
        p('rgb_encoding', 'raw')
        p('jpeg_quality', 90)
        p('depth_encoding', 'raw')
        p('use_roi', False)              # full frame by default; see realtime_fp.md S4
        p('roi_margin_px', 40)
        p('ego_motion', True)
        p('fit_tol_m', 0.010)
        p('fit_lost', 0.5)
        p('fit_recover', 0.6)
        p('lost_frames', 3)
        p('reseed', True)                # on LOST, re-register through register_url
        p('reseed_interval_sec', 3.0)
        p('request_timeout_sec', 60.0)
        p('mesh_marker', True)
        p('mesh_resource', 'package://admittance_control/models/{object}.ply')
        p('mesh_scale', 0.001)           # models/*.ply are mm
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._cfg = {n: g(n) for n in (
            'track_url', 'register_url', 'base_frame', 'child_frame', 'auto_start',
            'max_seed_age_sec', 'max_sync_delta_sec', 'max_in_flight', 'refine_iter',
            'rgb_encoding', 'jpeg_quality', 'depth_encoding', 'use_roi', 'roi_margin_px',
            'ego_motion', 'fit_tol_m', 'fit_lost', 'fit_recover', 'lost_frames', 'reseed',
            'reseed_interval_sec', 'request_timeout_sec', 'mesh_marker', 'mesh_resource',
            'mesh_scale')}

        self._lock = threading.Condition()
        self._running = True
        self._ws = None
        self._state = 'DISCONNECTED'   # DISCONNECTED / IDLE / TRACKING / LOST
        self._object = None
        self._seed = None              # last `start` sent, resent after a reconnect
        self._last_good = None         # (T_cam_obj, T_base_cam) of the last tracked frame
        self._K = None
        self._camera_frame = None
        self._rgb_q = deque(maxlen=6)
        self._depth_q = deque(maxlen=6)
        self._last_pair = None
        self._frames = fs.LatestSlot()
        self._in_flight = 0
        self._seq = 0
        self._sent = {}                # seq -> (header stamp msg, frame_id, T_base_cam)
        self._hint = None
        self._last_detection = None
        self._reseed_busy = False
        self._reseed_asked = 0.0
        self._reseed_now = False
        self._reseed_frames = {}       # seq -> (rgb, depth_mm, K, T_base_cam, stamp)
        self._stats = deque(maxlen=300)  # (wall time, delay s, fit)

        self._tf_buffer = Buffer(cache_time=Duration(seconds=120.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_pub = TransformBroadcaster(self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pose_pub = self.create_publisher(PoseStamped, g('pose_topic'), 10)
        self._status_pub = self.create_publisher(String, g('status_topic'), 10)
        self._marker_pub = self.create_publisher(Marker, g('marker_topic'), latched)
        self.create_subscription(Image, g('rgb_topic'), self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Detection3DArray, g('detections_topic'), self._on_detection,
                                 latched)
        self.create_service(Trigger, '~/start', self._srv_start)
        self.create_service(Trigger, '~/stop', self._srv_stop)
        self.create_service(Trigger, '~/reseed', self._srv_reseed)
        self.create_timer(0.5, self._publish_status)

        self._threads = [threading.Thread(target=self._send_loop, daemon=True),
                         threading.Thread(target=self._recv_loop, daemon=True)]
        for t in self._threads:
            t.start()
        self.get_logger().info(
            f"tracker at {self._cfg['track_url']}, re-registration at "
            f"{self._cfg['register_url']}, publishing {g('pose_topic')} + TF -> "
            f"{self._cfg['child_frame']}")

    # ── camera input ──────────────────────────────────────────────────────

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_rgb(self, msg: Image) -> None:
        self._rgb_q.append(msg)
        self._pair(msg, self._depth_q, rgb_first=True)

    def _on_depth(self, msg: Image) -> None:
        self._depth_q.append(msg)
        self._pair(msg, self._rgb_q, rgb_first=False)

    def _pair(self, msg, others, rgb_first):
        """Offer the newest RGB-D pair to the sender (latest wins). Pairing is by
        stamp: the camera node stamps both images of a frame identically."""
        if not others:
            return
        ns = stamp_ns(msg.header.stamp)
        other = min(others, key=lambda o: abs(stamp_ns(o.header.stamp) - ns))
        if abs(stamp_ns(other.header.stamp) - ns) > self._cfg['max_sync_delta_sec'] * 1e9:
            return
        if ns == self._last_pair:
            return
        self._last_pair = ns
        self._frames.put((msg, other) if rgb_first else (other, msg))

    def _lookup_base_cam(self, frame_id, stamp) -> Optional[np.ndarray]:
        try:
            t = self._tf_buffer.lookup_transform(self._cfg['base_frame'], frame_id,
                                                 Time.from_msg(stamp),
                                                 timeout=Duration(seconds=0.05))
            return transform_to_matrix(t)
        except Exception:  # noqa: BLE001 - no TF: track without ego-motion
            return None

    # ── seeding ───────────────────────────────────────────────────────────

    def _on_detection(self, msg: Detection3DArray) -> None:
        if not msg.detections or not msg.detections[0].results:
            return
        self._last_detection = msg
        age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if not self._cfg['auto_start']:
            return
        if age > self._cfg['max_seed_age_sec']:
            self.get_logger().info(f'ignoring a {age:.0f} s old detection as a seed '
                                   '(call ~/start to use it)')
            return
        ok, message = self._start_from_detection(msg)
        (self.get_logger().info if ok else self.get_logger().warn)(message)

    def _start_from_detection(self, msg: Detection3DArray):
        hyp = msg.detections[0].results[0]
        name = hyp.hypothesis.class_id or None
        T = pose_msg_to_matrix(hyp.pose.pose)
        # The pose belongs to the registered frame; TF at its stamp lets the tracker
        # cancel any robot motion since. (The bridge must stamp detections with the
        # captured frame's stamp for this to be exact.)
        T_base_cam = self._lookup_base_cam(msg.header.frame_id, msg.header.stamp)
        self._start(T, name, T_base_cam)
        return True, (f'tracking {name} from the bridge detection '
                      f'(t={np.round(T[:3, 3], 3).tolist()} m, '
                      f'{"with" if T_base_cam is not None else "without"} T_base_cam)')

    def _start(self, T_cam_obj, name, T_base_cam):
        c = self._cfg
        msg = {'type': 'start', 'seed': 'pose',
               'pose': np.asarray(T_cam_obj).reshape(-1).tolist(), 'object': name,
               'T_base_cam': None if T_base_cam is None else
               np.asarray(T_base_cam).reshape(-1).tolist(),
               'refine_iter': c['refine_iter'], 'ego_motion': c['ego_motion'],
               'fit_tol_m': c['fit_tol_m'], 'fit_lost': c['fit_lost'],
               'fit_recover': c['fit_recover'], 'lost_frames': c['lost_frames']}
        with self._lock:
            self._seed = msg
            self._object = name
            self._last_good = (np.asarray(T_cam_obj), T_base_cam)
        self._send_text(msg)
        self._publish_marker(name)

    def _send_text(self, obj) -> bool:
        ws = self._ws
        if ws is None:
            return False  # the sender loop resends self._seed once connected
        try:
            ws.send(json.dumps(obj))
            return True
        except Exception as exc:  # noqa: BLE001
            self._disconnected(f'send failed: {exc}')
            return False

    def _srv_start(self, request, response):
        if self._last_detection is None:
            response.success, response.message = False, 'no detection received yet'
            return response
        response.success, response.message = self._start_from_detection(self._last_detection)
        return response

    def _srv_stop(self, request, response):
        with self._lock:
            self._seed = None
        self._send_text({'type': 'stop'})
        response.success, response.message = True, 'tracking stopped'
        return response

    def _srv_reseed(self, request, response):
        with self._lock:
            if self._last_good is None:
                response.success, response.message = False, 'no pose to project a mask from'
                return response
            self._reseed_now = True
        response.success = True
        response.message = 're-registration requested on the next frame'
        return response

    # ── connection ────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            ws = websocket.create_connection(
                self._cfg['track_url'], timeout=10, enable_multithread=True,
                sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),))
        except Exception:  # noqa: BLE001 - tracker not up yet; retry quietly
            return False
        ws.settimeout(None)
        with self._lock:
            self._ws = ws
            self._state = 'IDLE'
            self._in_flight = 0
            self._sent.clear()
            seed = self._seed
            if seed is not None and self._last_good is not None:
                # Reconnected mid-session: continue from the last tracked pose.
                T, T_bc = self._last_good
                seed = dict(seed, pose=T.reshape(-1).tolist(),
                            T_base_cam=None if T_bc is None else T_bc.reshape(-1).tolist())
                self._seed = seed
        self.get_logger().info(f"connected to the tracker at {self._cfg['track_url']}")
        if seed is not None:
            self._send_text(seed)
        return True

    def _disconnected(self, why):
        with self._lock:
            ws, self._ws = self._ws, None
            was = self._state
            self._state = 'DISCONNECTED'
            self._in_flight = 0
            self._lock.notify_all()
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
            if was != 'DISCONNECTED' and self._running:
                self.get_logger().warn(f'tracker connection lost ({why}); reconnecting')

    # ── sender: newest frame, at most max_in_flight outstanding ───────────

    def _send_loop(self):
        while self._running:
            if self._ws is None:
                if not self._connect():
                    time.sleep(1.0)
                continue
            with self._lock:
                self._lock.wait_for(lambda: self._in_flight < self._cfg['max_in_flight']
                                    or not self._running or self._ws is None, timeout=0.5)
                if self._in_flight >= self._cfg['max_in_flight'] or self._ws is None:
                    continue
            item, _ = self._frames.get(timeout=0.5)
            if item is None:
                continue
            with self._lock:
                state = self._state
            if state not in ('TRACKING', 'LOST') or self._K is None:
                continue  # nothing to track yet: do not spend CPU encoding
            try:
                self._send_frame(*item, state)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not stop the loop
                self.get_logger().warn(f'could not send a frame: {exc}',
                                       throttle_duration_sec=5.0)

    def _send_frame(self, rgb_msg, depth_msg, state):
        c = self._cfg
        rgb = color_to_rgb(rgb_msg)
        depth = depth_to_mm(depth_msg)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f'rgb {rgb.shape[:2]} and depth {depth.shape} differ; '
                             'the depth must be aligned to colour')
        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id or 'camera_color_optical_frame'
        self._camera_frame = frame_id
        T_base_cam = self._lookup_base_cam(frame_id, stamp)
        K = self._K.copy()

        with self._lock:
            want_mask = self._reseed_now or (
                state == 'LOST' and c['reseed'] and not self._reseed_busy and
                time.monotonic() - self._reseed_asked >= c['reseed_interval_sec'])
            roi = None
            if c['use_roi'] and state == 'TRACKING' and not want_mask:
                roi = fs.roi_from_hint(self._hint, rgb.shape[:2], margin_px=c['roi_margin_px'])
            self._seq += 1
            seq = self._seq
            if want_mask:
                self._reseed_now = False
                self._reseed_busy = True
                self._reseed_asked = time.monotonic()
                self._reseed_frames[seq] = (rgb, depth, K, T_base_cam, stamp)

        msg = fs.pack_frame(seq, stamp.sec + stamp.nanosec * 1e-9, K, rgb.shape[:2],
                            fs.crop(rgb, roi), fs.crop(depth, roi), 0.001, roi=roi,
                            T_base_cam=T_base_cam, rgb_enc=c['rgb_encoding'],
                            depth_enc=c['depth_encoding'], jpeg_quality=c['jpeg_quality'],
                            want_reseed_mask=want_mask)
        ws = self._ws
        if ws is None:
            return
        with self._lock:
            self._sent[seq] = (stamp, frame_id, T_base_cam)
            self._in_flight += 1
        try:
            ws.send_binary(msg)
        except Exception as exc:  # noqa: BLE001
            self._disconnected(f'send failed: {exc}')

    # ── receiver: replies -> ROS ──────────────────────────────────────────

    def _recv_loop(self):
        while self._running:
            ws = self._ws
            if ws is None:
                time.sleep(0.05)
                continue
            try:
                raw = ws.recv()
            except Exception as exc:  # noqa: BLE001
                if self._running:
                    self._disconnected(f'receive failed: {exc}')
                continue
            if not raw:
                continue
            try:
                self._on_reply(json.loads(raw))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'bad reply from the tracker: {exc}',
                                       throttle_duration_sec=5.0)

    def _on_reply(self, reply):
        kind = reply.get('type')
        if kind == 'started':
            with self._lock:
                self._state = reply['state']
                self._object = reply.get('object_name')
            self.get_logger().info(f"tracker started on {reply.get('object_name')} "
                                   f"(diameter {reply.get('diameter_m', 0) * 1000:.0f} mm)")
            return
        if kind == 'stopped':
            with self._lock:
                self._state = 'IDLE'
            return
        if kind == 'error':
            with self._lock:
                self._in_flight = max(self._in_flight - 1, 0)
                self._lock.notify_all()
            self.get_logger().warn(f"tracker: {reply.get('message')}")
            return
        if kind != 'pose':
            return

        seq = reply['seq']
        with self._lock:
            self._in_flight = max(self._in_flight - 1 - reply.get('dropped', 0), 0)
            self._lock.notify_all()
            previous = self._state
            self._state = reply['state']
            self._hint = reply.get('hint')
            meta = self._sent.pop(seq, None)
            for old in [k for k in self._sent if k < seq]:  # dropped by the tracker
                del self._sent[old]
            reseed_frame = self._reseed_frames.pop(seq, None)
            for old in [k for k in self._reseed_frames if k < seq]:
                del self._reseed_frames[old]
                self._reseed_busy = False

        if reply['state'] != previous and previous in ('TRACKING', 'LOST'):
            (self.get_logger().warn if reply['state'] == 'LOST' else self.get_logger().info)(
                f"{self._object}: {previous} -> {reply['state']} (fit {reply.get('fit')})")
        if reply.get('error'):
            self.get_logger().warn(f"tracker step failed: {reply['error']}",
                                   throttle_duration_sec=5.0)

        if meta is not None and reply['state'] == 'TRACKING' and reply.get('pose'):
            stamp, frame_id, T_base_cam = meta
            T = np.asarray(reply['pose'], dtype=np.float64)
            self._publish_pose(T, stamp, frame_id)
            with self._lock:
                self._last_good = (T, T_base_cam)
            delay = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
            self._stats.append((time.time(), delay, reply.get('fit')))

        if reseed_frame is not None:
            threading.Thread(target=self._reseed, daemon=True,
                             args=(reseed_frame, reply.get('reseed_mask_png'))).start()

    def _reseed(self, frame, mask_b64):
        """LOST (or ~/reseed): this frame + the projected mask -> desktop register ->
        restart the tracker from the answer. Runs in its own thread; streaming and the
        tracker's own recovery attempts go on meanwhile."""
        rgb, depth, K, T_base_cam, stamp = frame
        try:
            if not mask_b64:
                raise RuntimeError('the tracker had no pose to project a mask from')
            T, name, sec = fs.post_predict_pose(
                self._cfg['register_url'], K, 0.001, rgb, depth, base64.b64decode(mask_b64),
                timeout=self._cfg['request_timeout_sec'])
            with self._lock:
                stopped = self._seed is None
            if stopped:
                return
            self._start(T, name, T_base_cam)
            self.get_logger().info(f're-registered {name} on the desktop in {sec:.1f} s; '
                                   'tracking restarted from it')
        except RuntimeError as err:
            self.get_logger().warn(f're-registration failed: {err}')
        finally:
            with self._lock:
                self._reseed_busy = False

    # ── publishing ────────────────────────────────────────────────────────

    def _publish_pose(self, T, stamp, frame_id):
        qx, qy, qz, qw = rotmat_to_quat(T[:3, :3])
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = \
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        pose.pose.orientation.x, pose.pose.orientation.y = float(qx), float(qy)
        pose.pose.orientation.z, pose.pose.orientation.w = float(qz), float(qw)
        self._pose_pub.publish(pose)

        tf = TransformStamped()
        tf.header = pose.header
        tf.child_frame_id = self._cfg['child_frame']
        tf.transform.translation.x = pose.pose.position.x
        tf.transform.translation.y = pose.pose.position.y
        tf.transform.translation.z = pose.pose.position.z
        tf.transform.rotation = pose.pose.orientation
        self._tf_pub.sendTransform(tf)

    def _publish_marker(self, name):
        if not self._cfg['mesh_marker'] or not name or not self._cfg['mesh_resource']:
            return
        m = Marker()
        m.header.frame_id = self._cfg['child_frame']
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'fp_object', 0
        m.type, m.action = Marker.MESH_RESOURCE, Marker.ADD
        m.mesh_resource = self._cfg['mesh_resource'].format(object=name)
        m.frame_locked = True  # redrawn at every TF update, no per-frame geometry
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = float(self._cfg['mesh_scale'])
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 0.9, 0.6
        self._marker_pub.publish(m)

    def _publish_status(self):
        now = time.time()
        recent = [s for s in self._stats if now - s[0] <= 2.0]
        with self._lock:
            state, obj, in_flight = self._state, self._object, self._in_flight
        fit = recent[-1][2] if recent else None
        status = {
            'state': state, 'object': obj,
            'rate_hz': round(len(recent) / 2.0, 1),
            'delay_ms': round(1000 * float(np.mean([s[1] for s in recent])), 1) if recent else None,
            'fit': fit, 'in_flight': in_flight, 'reseeding': self._reseed_busy,
        }
        self._status_pub.publish(String(data=json.dumps(status)))

    def shutdown(self):
        self._running = False
        self._frames.close()
        with self._lock:
            self._lock.notify_all()
        self._disconnected('shutdown')


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = FpTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
