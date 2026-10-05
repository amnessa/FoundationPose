# Live pose tracking with FoundationPose: plan

*Written 2026-10-02, after deciding that FoundationPose takes over live tracking from ICP
(`todo.md` → Decisions; `thesis_notes.md` 2026-10-02). Priority: **speed**, meaning high
pose rate and low delay. ICP keeps the stationary work: the `run_icp` seed, `save_object`,
`refine_pose`.*

## In simple terms

Today FoundationPose runs once per part:

1. we send one image;
2. it searches hundreds of guesses (`register`, ~1–2 s);
3. the ICP node then follows the part on the CPU.

FoundationPose has a second mode, `track_one`. It starts from the last pose and only nudges
it with a couple of passes of its refiner network on the GPU, which is milliseconds instead
of seconds. To see the part move live in RViz we have to:

- feed it every camera frame;
- get each pose back;
- publish it as TF;

all faster than the camera produces frames. The camera is on this laptop, but
FoundationPose runs in Docker on the GPU host (over Tailscale), so most of this plan is
about **not wasting time between the camera and the GPU**.

## Targets

| | minimum | goal |
|---|---|---|
| pose rate in RViz | 15 Hz | 30 Hz (the camera's rate) |
| delay, image stamp → TF published | ≤ 100 ms | ≤ 50 ms |
| `register` (seed and re-seed) | as today | — |

## Where the time goes, per frame

```
camera ─► ROS image ─► encode ─► network ─► decode ─► GPU: depth filter + 2 refiner passes ─► reply ─► TF
 (laptop)                      (Tailscale)               (GPU host or laptop)
```

Every stage is measured, not guessed: each reply carries the server's own timings
(receive, decode, GPU, send), and the client adds its own (image stamp, sent, reply).

## Design decisions for speed

**S1. Measure `track_one` alone first, on both GPUs, before writing any transport.**
- What it decides: whether tracking runs on the **laptop** (RTX 4060 8 GB, visible in this
  container) or on the **GPU host**.
- Why it matters: if the laptop tracks at ≥ 20 Hz by itself, there is no network in the
  loop at all, and that beats any transport. Registration (heavy: the scorer plus ~250
  hypotheses) can stay on the host either way.
- What to check for a local install:
  - FoundationPose's dependencies (nvdiffrast, pytorch3d, the `mycpp` extension) against
    the local torch 2.12;
  - otherwise the FoundationPose Docker image on the laptop, which needs the NVIDIA
    container runtime on the laptop host;
  - VRAM: tracking needs only the refiner network, so it should fit.

**S2. Tracking state lives on the server; the client only streams.**
- The server keeps the last pose (`est.pose_last`) and tracks each frame from it.
- So the client does not wait for pose N before sending frame N+1. Up to 2 frames in
  flight hides the network round trip.
- The server keeps **only the newest** waiting frame and drops older ones ("latest wins").
- Result: the rate is limited by the GPU, not by GPU + round trip, and the delay never
  grows into a backlog.

**S3. One persistent binary stream, not an HTTP request per frame.**
- **WebSocket**, binary messages, one TCP connection. Per frame there is a small JSON
  header (seq, stamp, K, depth scale, ROI, `T_base_cam`), then the RGB bytes, then the
  depth bytes. The reply is JSON: seq, stamp, pose 4×4, fit, timings.
- Why not HTTP: one request in flight per connection, with headers and multipart
  parsing every frame. The current Flask server is also `threaded=False` and writes files
  to disk per request.
- Why not gRPC: code generation for no gain here.
- Why not ROS 2 across Tailscale: DDS needs discovery work over Tailscale (Zenoh or a
  discovery server), and raw 1280×720 RGB-D is ~4.6 MB/frame, ~140 MB/s at 30 Hz.
- Alternative if WebSocket disappoints: ZeroMQ with `CONFLATE` (latest-wins built in).
- **The same protocol works locally** (`ws://localhost`), so S1's answer only changes a URL.

**S4. Send only the region around the part (ROI crop).**
- FoundationPose itself only looks at a square around the part:
  - centred on the projected pose;
  - radius = f · (mesh diameter × 1.2 / 2) / z;
  - resampled to 160×160.
- So the client crops the image to that square plus a motion margin (grows with the
  part's pixel speed, minimum ~40 px) and shifts K (`cx −= x0`, `cy −= y0`).
- Effect: full resolution where it matters, 5–20× fewer bytes, and a smaller depth filter
  on the GPU.
- If the pose is lost, send the full frame.

**S5. Cheap encoding, no disk.**
- Remote:
  - RGB as JPEG q90, a few ms with OpenCV;
  - depth as raw uint16, PNG at compression level 1 (lossless — never JPEG depth).
- Local: raw bytes, no encoding.
- The server decodes in memory, puts depth on the GPU in metres, and writes nothing to disk.
- No overlay rendering per frame; one optional debug image every N frames.

**S6. Fewer refiner passes, lighter mesh.**
- Try `track_one(iteration=1)` against the default 2. Accept 1 only if the jitter on a
  still part does not grow (S1 and stage 5 measure this).
- Decimate the CAD to a few thousand faces for rendering (plates need very few).
- Keep `debug=0`. Check that the refiner runs under autocast/fp16 (`amp`) and with
  `torch.backends.cudnn.benchmark`.

**S7. Cancel the robot's own motion (eye-in-hand).**
- The camera is on the wrist. When the robot moves, the part jumps in the image even
  though it is still, and a fast robot move looks like a fast part.
- Each frame carries `T_base_cam` from TF at the image stamp. Before `track_one`, the
  server predicts:

  `pose_pred = inv(T_base_cam_now) · T_base_cam_prev · pose_prev`

- The tracker then only sees the part's own motion. That is more robust and allows a lower
  rate while the arm moves.
- Gotcha: FoundationPose stores `pose_last` in its *centred-mesh* frame.
  - It returns `pose_last @ tf_to_centered_mesh`.
  - So to seed or predict, set `pose_last = T_cam_obj @ inv(tf_to_centered_mesh)`.
  - Check against `estimater.py`.

**S8. RViz gets a TF per frame, not geometry per frame.**
- Per frame, publish:
  - TF `camera_color_optical_frame → fp_object`, stamped with the **image** stamp, so
    RViz places it where the moving camera was;
  - `/perception/fp/pose` (PoseStamped).
- Once, a latched `Marker` MESH_RESOURCE of the CAD in frame `fp_object` with
  `frame_locked: true`: RViz redraws it at every TF update, and no point cloud is sent per
  frame.
- A status topic: rate, delay, fit, state.

**S9. Know when tracking is lost, cheaply.**
- `track_one` returns a pose but no score.
- So the server renders the CAD's depth at the new pose (in the 160×160 crop it already
  works in) and computes `fit`: the fraction of rendered pixels whose measured depth is
  within 10 mm. This is the same idea as the ICP node's fitness, at negligible GPU cost.
- After `fit` stays below a threshold (start 0.5) for 3 frames:
  1. state = LOST;
  2. the client stops publishing TF;
  3. it re-seeds automatically: a mask from the CAD projected at the last good pose
     (dilated) → `register()` on the newest frame, with no click needed;
  4. a SAM2 click is the fallback.

**S10. One part tracked at a time** (the one being placed; the saved ones are static).
- Several at once later: one FoundationPose instance per part, sharing the same
  refiner/scorer networks, each tracked in turn per frame (the cost grows linearly).

**S11. Registration and tracking share one GPU process.**
- The WebSocket server runs in a thread of `fp_server.py`, sharing `EST` and the loaded
  networks, with one GPU lock.
- A `register` pauses tracking, and the new pose becomes `pose_last` directly.
- Tracking can also be seeded with a pose the client sends: the ICP pose, or a saved part.

## Implementation steps, each with its check

0. **Benchmark `track_one` alone** — `scripts_in_foundationpose/track_bench.py`, run on
   the host and on the laptop.
   - Setup:
     - load the ear and base CADs;
     - `register` on the saved `rgb_depth_to_send/` frame;
     - then `track_one` 300×, on the full frame and on the ROI crop.
   - Report ms/frame (mean, p95) and GPU memory for:
     - `iteration` 1 / 2;
     - full vs decimated mesh;
     - full frame vs ROI.
   - *Gate:* pick the location (laptop if ≥ 20 Hz there). Write the numbers in
     `thesis_notes.md`.
1. **Network check** (only if remote):
   - `tailscale ping <host>`: a **direct** path, not "via DERP" (a relay adds tens of ms and
     caps throughput);
   - round trip and throughput with a dummy WebSocket echo of a 150 KB message.
   - *Gate:* RTT and transfer time per ROI frame both < 30 ms.
2. **Server tracking endpoint**: a WebSocket thread in `fp_server.py` (port 5001), in
   memory, with:
   - messages `start{pose | from_register}`, `frame{…}`, `stop`;
   - the S7 prediction, the S9 fit and per-stage timings in every reply;
   - the GPU lock with `/predict_pose`.
   - *Check:* `track_bench.py --ws` replays 300 saved frames through the socket and
     matches step 0's GPU time within a few ms.
3. **ROS client node**: `scripts/fp_tracker_node.py`.
   - Subscriptions: RGB and depth by exact stamp, plus `camera_info`.
   - A sender thread: latest-wins, ≤ 2 in flight, ROI crop + K shift, `T_base_cam` from
     TF.
   - Publishes the TF, pose, latched mesh marker and status (S8).
   - Services: `~/start` (seed: `last_registration` | `icp` | `pose`), `~/stop`.
   - Pure helpers (message packing, ROI/K, prediction, the latest-wins slot) in
     `admittance_control/fp_stream.py`, unit-tested without a GPU.
   - *Check:* live RViz with a still part, a part pushed by hand, and the robot moving with
     the part still (the TF must stay put in `base_link`). Status shows rate and delay
     against the targets.
4. **Lost and re-seed** (S9).
   - *Check:* cover the part with a hand and move it away: LOST within 3 frames, and an
     automatic re-seed when it reappears; a wrong part is never tracked silently.
5. **Measure against ICP and the stationary pose.**
   - `pose_jitter_probe.py` on `/perception/fp/pose` and `/perception/icp/refined_pose`:
     still for 30 s, and during a slow push. Compare noise, delay and drift.
   - The FP pose vs the `refine_pose` result at rest, which says how far tracking is from
     the 2.5 mm pipeline.
   - Results go to the README (new section) and `thesis_notes.md`.
6. **Hand-over to the assembly.**
   - The ICP node gets `tracking_source: icp | fp`. With `fp`:
     - its tracking timer is off (frees the CPU);
     - `current_pose` comes from `/perception/fp/pose`;
     - `save_object`'s rest detection and robust mean (`pose_stats`) work unchanged.
   - `refine_pose` stays the final stationary step.
   - *Check:* one full cycle: place → track → save → `refine_pose` → welding points →
     marks still within ~2.5 mm.

**Later, only if the targets are missed:**
- TensorRT for the refiner. NVIDIA's Isaac ROS FoundationPose already ships TensorRT
  tracking as a ROS 2 node; check its ROS distro and GPU needs before adopting it.
- Several parts tracked at once (S10).

## Parameters (`fp_tracker_node`, starting values)

| parameter | value |
|---|---|
| `track_url` | `ws://<host>:5001` |
| `max_in_flight` | 2 |
| `refine_iter` | 2 |
| `roi_margin_px` | 40 (min) |
| `rgb_encoding` | `jpeg` / `raw` |
| `jpeg_quality` | 90 |
| `depth_encoding` | `png1` / `raw` |
| `fit_tol_m` | 0.010 |
| `fit_lost` | 0.5 |
| `lost_frames` | 3 |
| `ego_motion` | true |
| `child_frame` | `fp_object` |
| `mesh_marker` | true |

## Open questions

- **Where the GPU host is:** the same LAN as the laptop, or remote (home)? This decides
  whether step 1 can pass at all, and makes local tracking (S1) more attractive.
- **The host's GPU model,** for comparing step 0's numbers.
- **The camera node's real rate:** it publishes 1280×720 RGB and aligned depth from Python
  at 30 Hz. Check with `ros2 topic hz`; if it falls short, it caps tracking before
  anything else does. 640×480 is the fallback, with S4 keeping detail at the part.
