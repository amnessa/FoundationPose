#!/usr/bin/env python3
"""Simulated tracking client: what fp_tracker_node will do, minus ROS.

Streams a saved frame to the tracking endpoint at the camera's rate, exactly the way
the ROS node is meant to (realtime_fp.md S2, S3, S5): never waits for a reply before
the next frame, at most --max-in-flight frames outstanding (extra camera frames are
skipped, not queued), JPEG rgb + lossless PNG depth, full frame (--roi to crop to the
server's hint instead). Only the pose comes back.

No torch, no ROS: needs numpy, OpenCV and websocket-client only, so it runs on the
host as well as in the container.

    # one machine: server in the container (--network=host), client anywhere here
    python3 scripts/track_replay.py
    # two machines: run this on the laptop, the server on the desktop
    python3 scripts/track_replay.py --url ws://<desktop>:5001
    # simulate motion: the image slides +-40 px at 0.5 Hz, the pose must follow
    python3 scripts/track_replay.py --wobble 40 --wobble-hz 0.5

Seeding (--seed): `register` (default) posts the frame + --mask to /predict_pose and
tracks from that registration, which is the real flow; `pose` uses a saved
detection_pem.json; `last` continues from whatever the server registered last.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_stream as fs  # noqa: E402

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("needs websocket-client: pip install websocket-client "
             "(Ubuntu: apt install python3-websocket)")

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(CODE_DIR, "Data", "Output", "foundationpose_results")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default=f"ws://127.0.0.1:{fs.DEFAULT_PORT}")
    p.add_argument("--frame-dir", default=os.path.join(CODE_DIR, "Data", "Input"))
    p.add_argument("--seed", choices=["register", "pose", "last"], default="register",
                   help="register: POST the frame + --mask to /predict_pose, then track "
                        "from that (the real flow); pose: --detection file; last: the "
                        "server's last registration")
    p.add_argument("--http-port", type=int, default=5000)
    p.add_argument("--mask", default=os.path.join(RESULTS, "mask.png"),
                   help="object mask for --seed register")
    p.add_argument("--detection", default=os.path.join(RESULTS, "detection_pem.json"),
                   help="seed pose + object name for --seed pose")
    p.add_argument("--rate", type=float, default=30.0, help="camera rate, Hz")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--max-in-flight", type=int, default=2)
    p.add_argument("--roi", action="store_true",
                   help="crop to the server's hint + margin (S4); default: full frame")
    p.add_argument("--margin", type=int, default=40, help="ROI motion margin, px")
    p.add_argument("--rgb-enc", choices=["jpeg", "raw"], default="jpeg")
    p.add_argument("--depth-enc", choices=["png", "raw"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--refine-iter", type=int, default=2)
    p.add_argument("--wobble", type=float, default=0.0,
                   help="slide the image by this many px (sine) to simulate motion")
    p.add_argument("--wobble-hz", type=float, default=0.5)
    return p.parse_args()


def load_frame(frame_dir):
    with open(os.path.join(frame_dir, "camera.json")) as fh:
        cam = json.load(fh)
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    scale_m = float(cam.get("depth_scale", 1.0)) / 1000.0
    rgb = cv2.cvtColor(cv2.imread(os.path.join(frame_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(frame_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
    return K, scale_m, rgb, depth


def register_over_http(args):
    """The frame + mask to /predict_pose, as the bridge node does. The tracker then
    starts from that registration (seed 'last')."""
    host = urllib.parse.urlparse(args.url).hostname
    url = f"http://{host}:{args.http_port}/predict_pose"
    boundary = uuid.uuid4().hex
    parts = []
    for field, path in (("rgb", "rgb.png"), ("depth", "depth.png"),
                        ("camera", "camera.json")):
        with open(os.path.join(args.frame_dir, path), "rb") as fh:
            parts.append((field, path, fh.read()))
    with open(args.mask, "rb") as fh:
        parts.append(("mask", "mask.png", fh.read()))
    body = b"".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{f}\"; "
        f"filename=\"{n}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
        + data + b"\r\n" for f, n, data in parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            reply = json.loads(resp.read())
    except urllib.error.HTTPError as err:
        sys.exit(f"/predict_pose failed: {err.code} {err.read()[:300]!r}")
    print(f"registered {reply.get('object_name')} via {url} in {time.time() - t:.1f} s "
          f"(score {reply.get('score', 0):.1f})")


def seed_message(args):
    seed = "last" if args.seed == "register" else args.seed
    msg = {"type": "start", "seed": seed, "refine_iter": args.refine_iter,
           # no robot here, so no T_base_cam: the wobble is "the part moving"
           "ego_motion": False}
    if args.seed == "pose":
        with open(args.detection) as fh:
            det = json.load(fh)[0]
        T = np.eye(4)
        T[:3, :3] = np.asarray(det["R"])
        T[:3, 3] = np.asarray(det["t"]) / 1000.0  # SAM-6D file: mm
        msg.update(pose=T.reshape(-1).tolist(), object=det.get("obj_name"))
    return msg


def shifted(img, dx, interp):
    if dx == 0:
        return img
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Client:
    def __init__(self, args):
        self.args = args
        self.ws = websocket.create_connection(
            args.url, timeout=60, enable_multithread=True,
            sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),))
        self.lock = threading.Lock()
        self.in_flight = 0
        self.hint = None
        self.state = "IDLE"
        self.replies = []      # (t_reply, reply, frame meta)
        self.meta = {}         # seq -> (stamp, bytes, encode_ms, roi, dx)
        self.errors = 0

    def start(self, msg):
        self.ws.send(json.dumps(msg))
        reply = json.loads(self.ws.recv())
        if reply.get("type") != "started":
            sys.exit(f"server refused to start: {reply}")
        self.state = reply["state"]
        print(f"tracking {reply['object_name']} (diameter {reply['diameter_m'] * 1000:.0f} mm, "
              f"seed {reply['seed']})")
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        while True:
            try:
                msg = self.ws.recv()
            except Exception:  # noqa: BLE001 - closed
                return
            t = time.time()
            reply = json.loads(msg)
            with self.lock:
                if reply.get("type") == "pose":
                    self.in_flight -= 1 + reply.get("dropped", 0)
                    self.in_flight = max(self.in_flight, 0)
                    self.state = reply["state"]
                    self.hint = reply.get("hint")
                    self.replies.append((t, reply, self.meta.pop(reply["seq"], None)))
                elif reply.get("type") == "error":
                    self.errors += 1
                    self.in_flight = max(self.in_flight - 1, 0)
                    print(f"server error: {reply.get('message')}")

    def send_frame(self, seq, stamp, K, rgb, depth, scale_m, dx):
        a = self.args
        with self.lock:
            if self.in_flight >= a.max_in_flight:
                return False
            roi = None
            if a.roi and self.state == "TRACKING":
                roi = fs.roi_from_hint(self.hint, rgb.shape[:2], margin_px=a.margin)
            self.in_flight += 1
        t = time.perf_counter()
        msg = fs.pack_frame(seq, stamp, K, rgb.shape[:2], fs.crop(rgb, roi),
                            fs.crop(depth, roi), scale_m, roi=roi, rgb_enc=a.rgb_enc,
                            depth_enc=a.depth_enc, jpeg_quality=a.jpeg_quality)
        enc_ms = (time.perf_counter() - t) * 1e3
        with self.lock:
            self.meta[seq] = (stamp, len(msg), enc_ms, roi, dx)
        self.ws.send_binary(msg)
        return True


def summarize(c, sent, skipped, seconds, u_ref):
    rows = [(t, r, m) for t, r, m in c.replies if m is not None]
    if not rows:
        print("no replies")
        return
    e2e = np.array([(t - m[0]) * 1e3 for t, r, m in rows])
    rtt = np.array([(t - r["t_sent"]) * 1e3 for t, r, m in rows])
    kb = np.array([m[1] / 1024 for t, r, m in rows])
    enc = np.array([m[2] for t, r, m in rows])
    tracked = [r for t, r, m in rows if r["state"] == "TRACKING"]
    fits = [r["fit"] for t, r, m in rows if r.get("fit") is not None]
    dropped = sum(r.get("dropped", 0) for t, r, m in rows)
    stages = {}
    for t, r, m in rows:
        for k, v in r["timings"].items():
            stages.setdefault(k, []).append(v)

    print(f"\n── summary ({seconds:.1f} s) " + "─" * 50)
    print(f"  frames: {sent} sent, {skipped} skipped by the client (in flight full), "
          f"{dropped} dropped by the server (latest wins), {len(rows)} answered")
    print(f"  pose rate      {len(tracked) / seconds:6.1f} Hz tracked "
          f"({len(rows) / seconds:.1f} Hz answered)")
    print(f"  stamp -> pose  mean {e2e.mean():6.1f}  p50 {np.percentile(e2e, 50):6.1f}  "
          f"p95 {np.percentile(e2e, 95):6.1f} ms   (target <= 100, goal <= 50)")
    print(f"  round trip     mean {rtt.mean():6.1f}  p95 {np.percentile(rtt, 95):6.1f} ms")
    print(f"  payload        mean {kb.mean():6.1f} KB/frame, client encode {enc.mean():.1f} ms")
    if fits:
        print(f"  fit            mean {np.mean(fits):.2f}  min {np.min(fits):.2f}")
    print("  server stages (mean ms): " + "  ".join(
        f"{k.replace('_ms', '')}={np.mean(v):.1f}" for k, v in stages.items()))
    if u_ref is not None:
        # The wobble moves the image by dx; a correct track moves its projection by dx.
        err = [abs(r["hint"]["u"] - (u_ref + m[4])) for t, r, m in rows
               if r.get("hint") and r["state"] == "TRACKING"]
        if err:
            print(f"  follow error   mean {np.mean(err):.1f} px  p95 "
                  f"{np.percentile(err, 95):.1f} px (projected centre vs applied shift)")


def main():
    args = parse_args()
    K, scale_m, rgb0, depth0 = load_frame(args.frame_dir)
    if args.seed == "register":
        register_over_http(args)
    c = Client(args)
    c.start(seed_message(args))

    period = 1.0 / args.rate
    still_s = 1.0 if args.wobble else 0.0   # settle before moving, to get u_ref
    sent = skipped = 0
    u_ref = None
    t0 = time.time()
    next_print = t0 + 2.0
    seq = 0
    while True:
        now = time.time()
        if now - t0 >= args.seconds:
            break
        tick = t0 + seq * period
        if tick > now:
            time.sleep(tick - now)
        stamp = time.time()  # "image stamp": when the camera produced the frame
        tm = stamp - t0 - still_s
        dx = args.wobble * np.sin(2 * np.pi * args.wobble_hz * tm) if tm > 0 else 0.0
        if args.wobble and u_ref is None and tm > 0:
            with c.lock:
                if c.hint and c.state == "TRACKING":
                    u_ref = c.hint["u"]
        rgb = shifted(rgb0, dx, cv2.INTER_LINEAR)
        depth = shifted(depth0, dx, cv2.INTER_NEAREST)
        if c.send_frame(seq, stamp, K, rgb, depth, scale_m, dx):
            sent += 1
        else:
            skipped += 1
        seq += 1
        if time.time() >= next_print:
            next_print += 2.0
            with c.lock:
                recent = [(t, r, m) for t, r, m in c.replies if t > time.time() - 2.0 and m]
                state = c.state
            if recent:
                lat = np.mean([(t - m[0]) * 1e3 for t, r, m in recent])
                fit = recent[-1][1].get("fit")
                roi = recent[-1][2][3]
                print(f"  {state:<8} {len(recent) / 2.0:5.1f} Hz  stamp->pose {lat:5.1f} ms  "
                      f"fit {fit if fit is None else round(fit, 2)}  "
                      f"{'roi ' + str(roi[2]) + 'x' + str(roi[3]) if roi else 'full frame'}")

    deadline = time.time() + 5.0
    while time.time() < deadline:
        with c.lock:
            if c.in_flight == 0:
                break
        time.sleep(0.01)
    seconds = time.time() - t0
    try:
        c.ws.send(json.dumps({"type": "stop"}))
        time.sleep(0.2)
        c.ws.close()
    except Exception:  # noqa: BLE001
        pass
    summarize(c, sent, skipped, seconds, u_ref)


if __name__ == "__main__":
    main()
