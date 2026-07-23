#!/usr/bin/env python3
"""FoundationPose bridge node.

Successor to ``sam6d_bridge_node.py``. Same job: capture a synchronized RGB-D
frame, POST it to the pose server, republish the result as a
``vision_msgs/Detection3DArray`` for the downstream ICP node. Same transfer
directory, same camera.json, same multipart request. What changed is on the far
side of the wire:

- The server no longer detects anything by itself. It opens the frame in a
  window on the *server* host, an operator clicks the object, and SAM2 turns
  that click into the mask FoundationPose registers against. So a trigger now
  blocks until someone clicks -- budget for that in ``request_timeout_sec``
  (default 300s, up from 120s).
- The reply carries one ``pose`` as a 4x4 matrix **in metres**, not a list of
  detections with millimetre translations. The /1000 conversion the SAM-6D
  bridge did is gone; re-introducing it would put the object 1000x too close.
- The reply also names the object. A PPF classifier on the server matches the
  masked cloud against its CAD library and picks the model FoundationPose then
  registers against, so the client no longer has to be told in advance which part
  is on the table. That name is republished here -- as ``class_id`` on the
  detection and on its own latched topic -- because the tracking node has to load
  the same .ply to run ICP against.

Trigger
-------
    ros2 service call /foundationpose_bridge/trigger std_srvs/srv/Trigger

Adding a CAD model at runtime
-----------------------------
When you compose a new part from existing models and export a .ply, the server
has to learn it before it can ever be classified. Point the node at the file and
call the upload service; the server indexes it, persists it into the library, and
it is classifiable from the next trigger onward -- no restart:

    ros2 param set /foundationpose_bridge model_ply_path /path/to/new_part.ply
    ros2 service call /foundationpose_bridge/add_model std_srvs/srv/Trigger

(A parameter plus ``Trigger`` rather than a custom service type, so that this
needs no additions to the package's CMakeLists and no rebuild of message
interfaces.)

Subscribes
----------
  <rgb_topic>    sensor_msgs/Image   (default /camera/color/image_raw)
  <depth_topic>  sensor_msgs/Image   (default /camera/depth/image_rect_raw)

Publishes
---------
  <detections_topic>   vision_msgs/Detection3DArray  (default /perception/detections)
  <object_name_topic>  std_msgs/String, latched       (default /perception/object_name)

  run from laptop: ros2 run admittance_control foundationpose_bridge_node.py \
      --ros-args -p server_url:=http://ip_to_home_desktop:5000/predict_pose
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from admittance_control.geometry import rotmat_to_quat
from admittance_control.sam6d_io import (
    encode_png,
    image_msg_to_numpy,
    load_camera_payload,
    normalize_color_image,
    normalize_depth_image,
    post_files,
    resolve_transfer_dir,
    save_artifacts,
)


def stamp_to_nanoseconds(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class PoseReply(NamedTuple):
    """One parsed ``/predict_pose`` reply."""
    ok: bool
    message: str
    pose: Optional[np.ndarray] = None
    score: float = 0.0
    artifacts: Dict[str, str] = {}
    object_name: str = ''
    classification: Dict[str, Any] = {}


def parse_pose_response(status_code: int, body: bytes) -> PoseReply:
    """Parse the FoundationPose ``/predict_pose`` reply.

    Success looks like ``{"status": "success", "pose": [[4x4]], "units": "m",
    "score": <float>, "object_name": "<cad stem>", "classification": {...},
    "artifacts": {<filename>: <base64>}}``.

    ``units`` is checked rather than assumed: the server is the one that decides
    the CAD scale, and a mesh accidentally left in millimetres is otherwise a
    silent 1000x error that only shows up as a wildly wrong pose.

    ``object_name`` is tolerated as absent, so this node still works against an
    older server that only ever knew one mesh.
    """
    text = body.decode('utf-8', errors='replace')
    if status_code != 200:
        return PoseReply(False, f'server returned HTTP {status_code}: {text[:500]}')
    try:
        payload: Any = json.loads(text)
    except ValueError:
        return PoseReply(False, f'response was not valid JSON: {text[:500]}')

    if not (isinstance(payload, dict) and payload.get('status') == 'success'):
        message = ''
        if isinstance(payload, dict):
            message = str(payload.get('message') or payload.get('error') or payload)
        return PoseReply(False, f'server reported failure: {message[:500]}')

    units = payload.get('units', 'm')
    if units != 'm':
        return PoseReply(False, f'server reported units "{units}"; this bridge expects metres')

    try:
        pose = np.asarray(payload['pose'], dtype=float).reshape(4, 4)
    except (KeyError, TypeError, ValueError) as exc:
        return PoseReply(False, f'could not read a 4x4 pose from the reply: {exc}')

    artifacts = payload.get('artifacts', {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    classification = payload.get('classification', {})
    if not isinstance(classification, dict):
        classification = {}
    score = float(payload.get('score', 0.0))
    name = str(payload.get('object_name') or '')

    summary = f'pose received (score {score:.3f})'
    if name:
        summary = f"'{name}' {summary}"
    return PoseReply(True, summary, pose, score, artifacts, name, classification)


class FoundationPoseBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('foundationpose_bridge')

        self.declare_parameter('server_url', 'http://127.0.0.1:5000/predict_pose')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('detections_topic', '/perception/detections')
        self.declare_parameter('object_name_topic', '/perception/object_name')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_id', '1')
        # The PPF classifier's answer is the name of a CAD file, which is far more
        # useful downstream than a fixed numeric id -- the tracking node has to load
        # that exact .ply. Set false to keep the old constant `object_id` in
        # class_id if something already depends on it.
        self.declare_parameter('use_ppf_name_as_class_id', True)
        # Set this, then call ~/add_model, to teach the server a new CAD file.
        self.declare_parameter('model_ply_path', '')
        self.declare_parameter('add_model_url', '')
        # Long by default: the server blocks on a human clicking the object.
        self.declare_parameter('request_timeout_sec', 300.0)
        self.declare_parameter('max_sync_delta_sec', 0.25)
        # FoundationPose's score is an unnormalized hypothesis-ranking score (tens,
        # not 0-1), unlike SAM-6D's probability. Leave this at 0 until you have
        # looked at the values your object actually produces.
        self.declare_parameter('min_score', 0.0)
        self.declare_parameter('auto_trigger', False)
        self.declare_parameter('results_dir', '')

        self._server_url = self.get_parameter('server_url').value
        self._request_timeout_sec = float(self.get_parameter('request_timeout_sec').value)
        self._max_sync_delta_sec = float(self.get_parameter('max_sync_delta_sec').value)
        self._camera_frame = self.get_parameter('camera_frame').value
        self._object_id = str(self.get_parameter('object_id').value)
        self._min_score = float(self.get_parameter('min_score').value)
        self._auto_trigger = bool(self.get_parameter('auto_trigger').value)
        self._use_ppf_name = bool(self.get_parameter('use_ppf_name_as_class_id').value)
        # Default the upload endpoint to /add_model on the same server, so only
        # `server_url` ever has to be configured.
        self._add_model_url = (str(self.get_parameter('add_model_url').value)
                               or self._server_url.rsplit('/', 1)[0] + '/add_model')

        self._transfer_dir = resolve_transfer_dir()
        self._camera_path = self._transfer_dir / 'camera.json'

        # The server's artifacts land here, overwritten each trigger:
        #   detection_pem.json  init pose, SAM-6D format (t in mm)   -> ICP node
        #   detection_ism.npz   segmentation (1,H,W) from SAM2       -> ICP node
        #   mask.png / vis_pose.png   the same result to look at
        # The ICP node reads the first two off disk rather than from the ROS
        # message, so point its `results_dir` parameter at THIS directory. If you
        # leave it on the old sam6d_results default it will happily refine against
        # a stale mask from a previous SAM-6D run.
        results_dir = str(self.get_parameter('results_dir').value)
        self._results_dir = (Path(results_dir) if results_dir
                             else self._transfer_dir.parent / 'foundationpose_results')

        self._latest_rgb: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._auto_done = False

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.create_subscription(Image, rgb_topic, self._on_rgb, 10)
        self.create_subscription(Image, depth_topic, self._on_depth, 10)

        # Latched so a late-joining ICP node still sees the last result.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._det_pub = self.create_publisher(
            Detection3DArray, self.get_parameter('detections_topic').value, latched)
        # Also latched, and for the same reason: a tracking node that starts after
        # the trigger still needs to know which .ply to load.
        self._name_pub = self.create_publisher(
            String, self.get_parameter('object_name_topic').value, latched)

        self._trigger_srv = self.create_service(Trigger, '~/trigger', self._on_trigger)
        self._add_model_srv = self.create_service(Trigger, '~/add_model', self._on_add_model)

        self.get_logger().info(
            f'FoundationPose bridge ready. RGB={rgb_topic} depth={depth_topic} '
            f'server={self._server_url} transfer_dir={self._transfer_dir} '
            f'results_dir={self._results_dir} '
            f'object_name_topic={self._name_pub.topic_name}')
        if self._auto_trigger:
            self.get_logger().info('auto_trigger=true: will run once on first synced pair.')

    # ── Subscriptions ────────────────────────────────────────────────────
    def _on_rgb(self, msg: Image) -> None:
        self._latest_rgb = msg
        self._maybe_auto_trigger()

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = msg
        self._maybe_auto_trigger()

    def _maybe_auto_trigger(self) -> None:
        if self._auto_trigger and not self._auto_done and self._pair_is_synced():
            self._auto_done = True
            ok, message, _ = self._run_once()
            log = self.get_logger().info if ok else self.get_logger().warn
            log(f'auto_trigger result: {message}')

    # ── Trigger service ──────────────────────────────────────────────────
    def _on_trigger(self, request, response):
        ok, message, _ = self._run_once()
        response.success = ok
        response.message = message
        return response

    # ── Upload a new CAD model to the server ─────────────────────────────
    def _on_add_model(self, request, response):
        """Send the .ply at ``model_ply_path`` to the server's /add_model.

        For the case where a part is composed from existing models and exported as
        a new mesh: the server has to index it before it can ever be classified.
        The parameter is read fresh on every call, so one `ros2 param set` plus one
        service call adds a model without restarting anything.
        """
        path = Path(str(self.get_parameter('model_ply_path').value))
        if not str(path):
            response.success = False
            response.message = ('set the model_ply_path parameter first, e.g. '
                                'ros2 param set /foundationpose_bridge model_ply_path '
                                '/path/to/new_part.ply')
            return response
        if not path.is_file():
            response.success = False
            response.message = f'no such file: {path}'
            return response

        try:
            payload = path.read_bytes()
            status_code, body = post_files(
                self._add_model_url,
                {'model': (path.name, payload, 'application/octet-stream')},
                self._request_timeout_sec)
        except Exception as exc:  # noqa: BLE001 - surface any HTTP/IO error
            response.success = False
            response.message = f'upload failed: {exc}'
            return response

        text = body.decode('utf-8', errors='replace')
        try:
            reply = json.loads(text)
        except ValueError:
            reply = {}
        if status_code != 200 or reply.get('status') != 'success':
            response.success = False
            response.message = (f'server refused the model (HTTP {status_code}): '
                                f'{reply.get("message", text[:300])}')
            return response

        # A wrong FreeCAD export unit surfaces here first, and it makes the model
        # unclassifiable rather than merely inaccurate, so it gets its own warning.
        if reply.get('scale_warning'):
            self.get_logger().error(f'SCALE: {reply["scale_warning"]}')

        similar = reply.get('similar_diameter') or []
        message = (f'server indexed \'{reply.get("object_name")}\'; library now holds '
                   f'{len(reply.get("models") or [])} model(s)')
        if reply.get('scale_warning'):
            message += f'. SCALE WARNING: {reply["scale_warning"]}'
        if similar:
            # The extent pre-filter cannot separate parts of near-identical size, so
            # these will now compete on verification score alone.
            message += (f'. Similar in size to {similar} -- watch the classification '
                        f'margin when any of them is the target')
        self.get_logger().info(message)
        response.success = True
        response.message = message
        return response

    # ── Sync check ───────────────────────────────────────────────────────
    def _pair_is_synced(self) -> bool:
        if self._latest_rgb is None or self._latest_depth is None:
            return False
        delta = abs(stamp_to_nanoseconds(self._latest_rgb)
                    - stamp_to_nanoseconds(self._latest_depth)) / 1e9
        return delta <= self._max_sync_delta_sec

    # ── One capture → POST → publish cycle ───────────────────────────────
    def _run_once(self):
        if self._latest_rgb is None or self._latest_depth is None:
            return False, 'no RGB-D frames received yet', 0
        if not self._pair_is_synced():
            return False, 'latest RGB and depth are not time-synchronized', 0

        rgb_msg, depth_msg = self._latest_rgb, self._latest_depth
        try:
            camera_payload = load_camera_payload(self._camera_path)
            rgb_image = normalize_color_image(image_msg_to_numpy(rgb_msg), rgb_msg.encoding)
            depth_image, resolved_camera = normalize_depth_image(
                image_msg_to_numpy(depth_msg), depth_msg.encoding, camera_payload)

            rgb_png = encode_png(rgb_image, 'rgb image')
            depth_png = encode_png(depth_image, 'depth image')
            camera_json = json.dumps(resolved_camera, separators=(',', ':'))

            self._transfer_dir.mkdir(parents=True, exist_ok=True)
            (self._transfer_dir / 'rgb.png').write_bytes(rgb_png)
            (self._transfer_dir / 'depth.png').write_bytes(depth_png)
            self._camera_path.write_text(camera_json, encoding='utf-8')

            files = {
                'rgb': ('rgb.png', rgb_png, 'image/png'),
                'depth': ('depth.png', depth_png, 'image/png'),
                'camera': ('camera.json', camera_json.encode('utf-8'), 'application/json'),
            }
            self.get_logger().info(
                'Sent frame to the FoundationPose server; waiting for the operator '
                'to click the object there...')
            status_code, body = post_files(self._server_url, files, self._request_timeout_sec)
        except Exception as exc:  # noqa: BLE001 - surface any capture/HTTP error
            return False, f'capture/POST failed: {exc}', 0

        reply = parse_pose_response(status_code, body)
        if not reply.ok:
            return False, reply.message, 0

        if reply.artifacts:
            try:
                written = save_artifacts(reply.artifacts, self._results_dir)
                self.get_logger().info(
                    f'saved {len(written)} artifact(s) to {self._results_dir}: {written}')
            except Exception as exc:  # noqa: BLE001 - artifact write is best-effort
                self.get_logger().warn(f'failed to save artifacts: {exc}')

        self._log_classification(reply)

        if reply.score < self._min_score:
            return False, (f'pose score {reply.score:.3f} below min_score '
                           f'{self._min_score:.3f}'), 0

        frame_id = rgb_msg.header.frame_id or self._camera_frame
        msg = self._build_detection_array(reply, frame_id)
        self._det_pub.publish(msg)
        # Published before the log line so a subscriber that reacts to the name is
        # not racing the pose it belongs to.
        if reply.object_name:
            self._name_pub.publish(String(data=reply.object_name))

        summary = (f'published {reply.object_name or "pose"} (score {reply.score:.3f}, '
                   f't={np.round(reply.pose[:3, 3], 4).tolist()} m) '
                   f'on {self._det_pub.topic_name}')
        self.get_logger().info(summary)
        return True, summary, len(msg.detections)

    def _log_classification(self, reply: PoseReply) -> None:
        """Surface what PPF decided, and how close the call was.

        The margin is the number to watch: two parts of similar size and shape are
        genuinely hard to separate from one viewpoint, and a thin margin is the
        server telling you so rather than a defect. If they turn out to be confused
        often, set PPF_MIN_MARGIN and PPF_STRICT on the server so it refuses instead
        of guessing.
        """
        c = reply.classification
        if not c:
            return
        scores = c.get('scores') or {}
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        table = ', '.join(f'{n}={s:.3f}' for n, s in top)
        line = (f"PPF chose '{reply.object_name}' "
                f"(margin {c.get('margin', 0.0):.3f} over {c.get('runner_up')}): {table}")
        if c.get('ambiguous'):
            self.get_logger().warn(line + '  -- flagged AMBIGUOUS by the server')
        else:
            self.get_logger().info(line)

    # ── pose → vision_msgs ───────────────────────────────────────────────
    def _build_detection_array(self, reply: PoseReply, frame_id: str) -> Detection3DArray:
        pose, score = reply.pose, reply.score
        # The CAD name if the server classified one, else the configured constant.
        # A tracking node reading class_id can then load exactly the mesh that was
        # registered, instead of being told out of band.
        class_id = (reply.object_name
                    if (self._use_ppf_name and reply.object_name) else self._object_id)

        out = Detection3DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id

        d = Detection3D()
        d.header = out.header
        d.id = class_id

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_id
        hyp.hypothesis.score = score
        # FoundationPose returns object-in-camera as a 4x4 in metres (the server
        # scales the CAD to metres on load), so the translation goes straight in.
        hyp.pose.pose.position.x = float(pose[0, 3])
        hyp.pose.pose.position.y = float(pose[1, 3])
        hyp.pose.pose.position.z = float(pose[2, 3])
        qx, qy, qz, qw = rotmat_to_quat(pose[:3, :3])
        hyp.pose.pose.orientation.x = qx
        hyp.pose.pose.orientation.y = qy
        hyp.pose.pose.orientation.z = qz
        hyp.pose.pose.orientation.w = qw
        d.results.append(hyp)

        # Centre the box on the object pose; no 3D extent is reported (the server
        # knows the CAD extents, the bridge does not).
        bbox = BoundingBox3D()
        bbox.center = hyp.pose.pose
        d.bbox = bbox

        out.detections.append(d)
        return out


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = FoundationPoseBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
