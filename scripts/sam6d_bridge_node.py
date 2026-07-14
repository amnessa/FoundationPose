#!/usr/bin/env python3
"""SAM-6D bridge node.

Subscribes to an RGB-D stream (simulation or real camera), and on request
captures a synchronized frame, sends it to the SAM-6D Flask server, parses the
6D-pose reply, and republishes it as a ``vision_msgs/Detection3DArray`` for the
downstream ICP node.

Trigger
-------
Call the ``~/trigger`` service (``std_srvs/Trigger``) to run one capture →
POST → publish cycle::

    ros2 service call /sam6d_bridge/trigger std_srvs/srv/Trigger

Set the ``auto_trigger`` parameter to true to run one cycle automatically as
soon as a synchronized RGB-D pair arrives (handy for quick tests).

Subscribes
----------
  <rgb_topic>    sensor_msgs/Image   (default /camera/color/image_raw)
  <depth_topic>  sensor_msgs/Image   (default /camera/depth/image_rect_raw)

Publishes
---------
  <detections_topic>  vision_msgs/Detection3DArray  (default /perception/detections)

Notes
-----
- Intrinsics come from ``camera.json`` in the transfer directory (the SAM-6D
  server needs cam_K + depth_scale); this mirrors the proven capture flow.
- SAM-6D translations are millimetres in the camera optical frame; they are
  converted to metres here. Rotations arrive as a 3x3 matrix (``R``).

  run from laptop ros2 run admittance_control sam6d_bridge_node.py   --ros-args -p server_url:=http://ip_to_home_desktop:5000/predict_pose
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
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
    parse_pem_response,
    post_files,
    resolve_transfer_dir,
    save_artifacts,
)


def stamp_to_nanoseconds(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class Sam6DBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('sam6d_bridge')

        self.declare_parameter('server_url', 'http://127.0.0.1:5000/predict_pose')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('detections_topic', '/perception/detections')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('request_timeout_sec', 120.0)
        self.declare_parameter('max_sync_delta_sec', 0.25)
        self.declare_parameter('min_score', 0.0)
        self.declare_parameter('auto_trigger', False)
        self.declare_parameter('results_dir', '')

        self._server_url = self.get_parameter('server_url').value
        self._request_timeout_sec = float(self.get_parameter('request_timeout_sec').value)
        self._max_sync_delta_sec = float(self.get_parameter('max_sync_delta_sec').value)
        self._camera_frame = self.get_parameter('camera_frame').value
        self._min_score = float(self.get_parameter('min_score').value)
        self._auto_trigger = bool(self.get_parameter('auto_trigger').value)

        self._transfer_dir = resolve_transfer_dir()
        self._camera_path = self._transfer_dir / 'camera.json'

        # Raw pipeline artifacts returned by the server are written here,
        # overwritten each trigger. Kept separate from the transfer dir, which
        # holds the rgb/depth/camera inputs we send out.
        results_dir = str(self.get_parameter('results_dir').value)
        self._results_dir = (Path(results_dir) if results_dir
                             else self._transfer_dir.parent / 'sam6d_results')

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

        self._trigger_srv = self.create_service(Trigger, '~/trigger', self._on_trigger)

        self.get_logger().info(
            f'SAM-6D bridge ready. RGB={rgb_topic} depth={depth_topic} '
            f'server={self._server_url} transfer_dir={self._transfer_dir} '
            f'results_dir={self._results_dir}')
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
            self.get_logger().info('Sending frame to SAM-6D server...')
            status_code, body = post_files(self._server_url, files, self._request_timeout_sec)
        except Exception as exc:  # noqa: BLE001 - surface any capture/HTTP error
            return False, f'capture/POST failed: {exc}', 0

        ok, message, raw_detections, artifacts = parse_pem_response(status_code, body)
        if not ok:
            return False, message, 0

        if artifacts:
            try:
                written = save_artifacts(artifacts, self._results_dir)
                self.get_logger().info(
                    f'saved {len(written)} artifact(s) to {self._results_dir}: {written}')
            except Exception as exc:  # noqa: BLE001 - artifact write is best-effort
                self.get_logger().warn(f'failed to save artifacts: {exc}')

        frame_id = rgb_msg.header.frame_id or self._camera_frame
        msg = self._build_detection_array(raw_detections, frame_id)
        self._det_pub.publish(msg)
        published = len(msg.detections)
        summary = f'published {published} detection(s) on {self._det_pub.topic_name}'
        self.get_logger().info(summary)
        return True, summary, published

    # ── detection_pem → vision_msgs ──────────────────────────────────────
    def _build_detection_array(self, raw_detections: List[dict],
                               frame_id: str) -> Detection3DArray:
        out = Detection3DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id

        for idx, det in enumerate(raw_detections):
            score = float(det.get('score', 0.0))
            if score < self._min_score:
                continue
            R = det.get('R')
            t = det.get('t')
            if R is None or t is None:
                continue

            d = Detection3D()
            d.header = out.header
            d.id = str(det.get('category_id', idx))

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det.get('category_id', ''))
            hyp.hypothesis.score = score
            # SAM-6D translation is millimetres in the camera optical frame.
            hyp.pose.pose.position.x = float(t[0]) / 1000.0
            hyp.pose.pose.position.y = float(t[1]) / 1000.0
            hyp.pose.pose.position.z = float(t[2]) / 1000.0
            qx, qy, qz, qw = rotmat_to_quat(R)
            hyp.pose.pose.orientation.x = qx
            hyp.pose.pose.orientation.y = qy
            hyp.pose.pose.orientation.z = qz
            hyp.pose.pose.orientation.w = qw
            d.results.append(hyp)

            # Only a 2D bbox is available from SAM-6D; place the box centre at
            # the object pose and leave size zero (no 3D extent estimated yet).
            bbox = BoundingBox3D()
            bbox.center = hyp.pose.pose
            d.bbox = bbox

            out.detections.append(d)

        return out


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = Sam6DBridgeNode()
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
