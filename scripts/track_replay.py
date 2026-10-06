#!/usr/bin/env python3
"""Simulated tracking client: what fp_tracker_node will do, minus ROS.

The split it simulates: registration on the desktop (fp_server.py /predict_pose,
one frame per part), tracking on the laptop (fp_track_server.py, every frame, over
localhost). Only the registration frames cross the network.

    1. POST one frame + mask to --register-url (/predict_pose)  -> pose, object name
    2. `start` the tracker at --url with that pose
    3. stream frames at the camera's rate: never wait for a reply before the next
       frame, at most --max-in-flight outstanding (extra camera frames are skipped,
       not queued), full frame (--roi to crop to the server's hint)
    4. on LOST: ask the tracker for a mask on one frame (`want_reseed_mask`), POST that
       frame + mask to /predict_pose, `start` again from the answer. Streaming goes on
       meanwhile; the tracker may also recover by itself.

No torch, no ROS: numpy, OpenCV and websocket-client only, so it runs on the host.

    # one machine, both roles (two processes in the laptop container):
    #   REGISTER_CROP=1 TRACK_ENABLE=0 PPF_ENABLE=0 python scripts/fp_server.py
    #   python scripts/fp_track_server.py
    python3 scripts/track_replay.py
    # the real split: tracker local, registration on the desktop
    python3 scripts/track_replay.py --register-url http://<desktop>:5000/predict_pose
    # motion, a covered part, a forced re-seed through the registration server
    python3 scripts/track_replay.py --wobble 40 --blackout 5 1.5 --force-reseed-at 10

Seeding (--seed): `register` (default) as above; `pose` uses a saved
detection_pem.json instead of step 1; `last` continues from the tracker's own last
registration (only when the tracker *is* fp_server.py).
"""

import argparse
import base64
import json
import os
import socket
import sys
import threading
import time

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
    p.add_argument("--url", default=f"ws://127.0.0.1:{fs.DEFAULT_PORT}",
                   help="the tracker (fp_track_server.py, or fp_server.py's endpoint)")
    p.add_argument("--register-url", default="http://127.0.0.1:5000/predict_pose",
                   help="the registration server's /predict_pose (the desktop)")
    p.add_argument("--frame-dir", default=os.path.join(CODE_DIR, "Data", "Input"))
    p.add_argument("--seed", choices=["register", "pose", "last"], default="register")
    p.add_argument("--mask", default=os.path.join(RESULTS, "mask.png"),
                   help="object mask for the first registration")
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
    p.add_argument("--no-reseed", action="store_true",
                   help="on LOST, do not re-register through --register-url")
    p.add_argument("--reseed-interval", type=float, default=3.0,
                   help="seconds between re-registration attempts")
    p.add_argument("--wobble", type=float, default=0.0,
                   help="slide the image by this many px (sine) to simulate motion")
    p.add_argument("--wobble-hz", type=float, default=0.5)
    p.add_argument("--blackout", type=float, nargs=2, metavar=("START", "SECONDS"),
                   help="blank rgb + depth for a while: the part is 'covered' -> LOST")
    p.add_argument("--force-reseed-at", type=float, metavar="SECONDS",
                   help="re-register once at this time even while tracking, to "
                        "exercise the whole re-seed path")
    return p.parse_args()


def load_frame(frame_dir):
    with open(os.path.join(frame_dir, "camera.json")) as fh:
        cam = json.load(fh)
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    scale_m = float(cam.get("depth_scale", 1.0)) / 1000.0
    rgb = cv2.cvtColor(cv2.imread(os.path.join(frame_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(frame_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
    return K, scale_m, rgb, depth


def start_message(args, pose=None, object_name=None):
    msg = {"type": "start", "refine_iter": args.refine_iter,
           # no robot here, so no T_base_cam: the wobble is "the part moving"
           "ego_motion": False}
    if pose is None:
        msg["seed"] = "last"
    else:
        msg.update(seed="pose", pose=np.asarray(pose).reshape(-1).tolist(),
                   object=object_name)
    return msg


def detection_pose(path):
    with open(path) as fh:
        det = json.load(fh)[0]
    T = np.eye(4)
    T[:3, :3] = np.asarray(det["R"])
    T[:3, 3] = np.asarray(det["t"]) / 1000.0  # SAM-6D file: mm
    return T, det.get("obj_name")


def shifted(img, dx, interp):
    if dx == 0:
        return img
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Client:
    def __init__(self, args, K, scale_m):
        self.args, self.K, self.scale_m = args, K, scale_m
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
        self.can_register = None
        # re-seeding through the registration server
        self.reseed_frames = {}  # seq -> (rgb, depth) of a frame we asked a mask for
        self.reseed_busy = False
        self.reseed_asked = 0.0
        self.reseeds = []        # (ok, seconds, message)

    def start(self, msg):
        """The first `start`, synchronously, before the receiver thread runs."""
        self.ws.send(json.dumps(msg))
        reply = json.loads(self.ws.recv())
        if reply.get("type") != "started":
            sys.exit(f"tracker refused to start: {reply}")
        self._started(reply)
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _started(self, reply):
        self.state = reply["state"]
        self.can_register = reply.get("can_register")
        print(f"tracking {reply['object_name']} (diameter {reply['diameter_m'] * 1000:.0f} mm, "
              f"seed {reply['seed']}, tracker can register: {reply.get('can_register')})")

    def _recv_loop(self):
        while True:
            try:
                msg = self.ws.recv()
            except Exception:  # noqa: BLE001 - closed
                return
            if not msg:
                continue
            t = time.time()
            reply = json.loads(msg)
            kind = reply.get("type")
            with self.lock:
                if kind == "pose":
                    self.in_flight = max(self.in_flight - 1 - reply.get("dropped", 0), 0)
                    self.state = reply["state"]
                    self.hint = reply.get("hint")
                    self.replies.append((t, reply, self.meta.pop(reply["seq"], None)))
                    frame = self.reseed_frames.pop(reply["seq"], None)
                elif kind == "started":
                    self._started(reply)
                    continue
                elif kind == "error":
                    self.errors += 1
                    self.in_flight = max(self.in_flight - 1, 0)
                    print(f"tracker error: {reply.get('message')}")
                    continue
                else:
                    continue
            if frame is not None:
                threading.Thread(target=self._reseed, daemon=True,
                                 args=(frame, reply.get("reseed_mask_png"))).start()

    def want_reseed(self):
        """Should the next frame carry `want_reseed_mask`? (LOST, or forced)"""
        with self.lock:
            stale = time.time() - self.reseed_asked > max(self.args.reseed_interval, 5.0)
            return not self.reseed_busy or stale

    def _reseed(self, frame, mask_b64):
        rgb, depth = frame
        try:
            if not mask_b64:
                raise RuntimeError("tracker had no good pose to project a mask from")
            pose, name, sec = fs.post_predict_pose(self.args.register_url, self.K, self.scale_m,
                                           rgb, depth, base64.b64decode(mask_b64))
            self.ws.send(json.dumps(start_message(self.args, pose, name)))
            self.reseeds.append((True, sec, name))
            print(f"  re-seed: registered {name} in {sec:.1f} s, tracker restarted")
        except RuntimeError as err:
            self.reseeds.append((False, 0.0, str(err)))
            print(f"  re-seed failed: {err}")
        finally:
            with self.lock:
                self.reseed_busy = False

    def send_frame(self, seq, stamp, rgb, depth, dx, want_mask=False):
        a = self.args
        with self.lock:
            if self.in_flight >= a.max_in_flight:
                return False
            roi = None
            if a.roi and self.state == "TRACKING" and not want_mask:
                roi = fs.roi_from_hint(self.hint, rgb.shape[:2], margin_px=a.margin)
            self.in_flight += 1
            if want_mask:
                self.reseed_busy = True
                self.reseed_asked = time.time()
                self.reseed_frames[seq] = (rgb, depth)
        t = time.perf_counter()
        msg = fs.pack_frame(seq, stamp, self.K, rgb.shape[:2], fs.crop(rgb, roi),
                            fs.crop(depth, roi), self.scale_m, roi=roi, rgb_enc=a.rgb_enc,
                            depth_enc=a.depth_enc, jpeg_quality=a.jpeg_quality,
                            want_reseed_mask=want_mask)
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
    states = {}
    for t, r, m in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    tracked = states.get("TRACKING", 0)
    fits = [r["fit"] for t, r, m in rows if r.get("fit") is not None]
    dropped = sum(r.get("dropped", 0) for t, r, m in rows)
    stages = {}
    for t, r, m in rows:
        for k, v in r["timings"].items():
            stages.setdefault(k, []).append(v)

    print(f"\n── summary ({seconds:.1f} s) " + "─" * 50)
    print(f"  frames: {sent} sent, {skipped} skipped by the client (in flight full), "
          f"{dropped} dropped by the tracker (latest wins), {len(rows)} answered")
    print(f"  states         " + ", ".join(f"{k} {v}" for k, v in states.items()))
    print(f"  pose rate      {tracked / seconds:6.1f} Hz tracked "
          f"({len(rows) / seconds:.1f} Hz answered)")
    print(f"  stamp -> pose  mean {e2e.mean():6.1f}  p50 {np.percentile(e2e, 50):6.1f}  "
          f"p95 {np.percentile(e2e, 95):6.1f} ms   (target <= 100, goal <= 50)")
    print(f"  round trip     mean {rtt.mean():6.1f}  p95 {np.percentile(rtt, 95):6.1f} ms")
    print(f"  payload        mean {kb.mean():6.1f} KB/frame, client encode {enc.mean():.1f} ms")
    if fits:
        print(f"  fit            mean {np.mean(fits):.2f}  min {np.min(fits):.2f}")
    print("  tracker stages (mean ms): " + "  ".join(
        f"{k.replace('_ms', '')}={np.mean(v):.1f}" for k, v in stages.items()))
    if c.reseeds:
        ok = [s for s in c.reseeds if s[0]]
        print(f"  re-seeds       {len(ok)} ok / {len(c.reseeds)} tried"
              + (f", register {np.mean([s[1] for s in ok]):.1f} s each" if ok else ""))
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
        with open(args.mask, "rb") as fh:
            mask_png = fh.read()
        try:
            pose, name, sec = fs.post_predict_pose(args.register_url, K, scale_m, rgb0, depth0, mask_png)
        except RuntimeError as err:
            sys.exit(str(err))
        print(f"registered {name} via {args.register_url} in {sec:.1f} s")
        first = start_message(args, pose, name)
    elif args.seed == "pose":
        first = start_message(args, *detection_pose(args.detection))
    else:
        first = start_message(args)

    c = Client(args, K, scale_m)
    c.start(first)

    period = 1.0 / args.rate
    still_s = 1.0 if args.wobble else 0.0   # settle before moving, to get u_ref
    sent = skipped = 0
    u_ref = None
    forced = args.force_reseed_at is None
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
        el = stamp - t0
        tm = el - still_s
        dx = args.wobble * np.sin(2 * np.pi * args.wobble_hz * tm) if tm > 0 else 0.0
        if args.wobble and u_ref is None and tm > 0:
            with c.lock:
                if c.hint and c.state == "TRACKING":
                    u_ref = c.hint["u"]
        if args.blackout and args.blackout[0] <= el < sum(args.blackout):
            rgb, depth = np.zeros_like(rgb0), np.zeros_like(depth0)
        else:
            rgb = shifted(rgb0, dx, cv2.INTER_LINEAR)
            depth = shifted(depth0, dx, cv2.INTER_NEAREST)

        force_now = not forced and el >= args.force_reseed_at
        want = force_now or (c.state == "LOST" and not args.no_reseed and
                             time.time() - c.reseed_asked >= args.reseed_interval and
                             c.want_reseed())
        if c.send_frame(seq, stamp, rgb, depth, dx, want_mask=want):
            sent += 1
            forced = forced or force_now
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
                print(f"  {el:5.1f}s {state:<8} {len(recent) / 2.0:5.1f} Hz  "
                      f"stamp->pose {lat:5.1f} ms  fit {fit if fit is None else round(fit, 2)}  "
                      f"{'roi ' + str(roi[2]) + 'x' + str(roi[3]) if roi else 'full frame'}")

    deadline = time.time() + 5.0
    while time.time() < deadline:
        with c.lock:
            if c.in_flight == 0 and not c.reseed_busy:
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
