# Host-side perception: SAM2 → PPF → FoundationPose

This document follows **one RGB-D frame from the camera through every step on the
server**. For each step it states three things: **what goes in** (with shape and
units), **what is done to it**, and **what comes out**. Plumbing (how a trigger is
fired, ROS details) comes last. The algorithm comes first.

> Written after the 1 October 2026 meeting with my advisor, where parts of the
> pipeline (especially mask → point cloud and the inside of PPF) were not
> explained clearly. Short answers to the questions raised there are in
> [Answers to the meeting questions](#answers-to-the-meeting-questions), including
> corrections to numbers I got wrong in the meeting.

---

## At a glance

```
 LAPTOP (robot side)                           DESKTOP PC (GPU, Docker, this repo)
 ───────────────────                           ───────────────────────────────────
 camera ─▶ foundationpose_bridge_node
             │  HTTP POST:
             │   rgb.png      1280×720×3  uint8
             │   depth.png    1280×720    uint16 (mm)
             │   camera.json  K (3×3) + depth_scale
             │   [click]      (u,v) pixel, optional
             ▼
                                         fp_server.py  :5000
                                         ┌──────────────────────────────────────┐
                                     0.  │ decode PNGs, depth → metres          │
                                         └──────────────────┬───────────────────┘
                                         ┌──────────────────┴───────────────────┐
                                     1.  │ SAM2 (looks at RGB only)             │
                                         │ click → 1280×720 binary mask         │
                                         └──────────────────┬───────────────────┘
                                         ┌──────────────────┴───────────────────┐
                                     2.  │ mask ⊙ depth → back-project with K   │
                                         │ → scene cloud  N×6, N ≤ 2000         │
                                         └──────────────────┬───────────────────┘
                                         ┌──────────────────┴───────────────────┐
                                     3.  │ PPF classifier                       │
                                         │ scene cloud vs. CAD library          │
                                         │ → a NAME only: "test_objv2_ear"      │
                                         └──────────────────┬───────────────────┘
                                         ┌──────────────────┴───────────────────┐
                                     4.  │ FoundationPose.register()            │
                                         │ RGB + depth + mask + chosen .ply     │
                                         │ → T (4×4), camera frame, metres      │
                                         └──────────────────┬───────────────────┘
             ◀── JSON: object_name, pose (4×4, m), score table, artifacts
             ▼
   publishes /perception/detections + /perception/object_name
             ▼
   tracking on the laptop, from T:
     • ICP node: loads the same .ply, snaps it to T, tracks it on the point cloud
     • FoundationPose track_one on the laptop GPU (new, see Live tracking)
```

Preparation (offline, once per new part) is a separate job:

```
 FreeCAD → .ply (triangle mesh) ─▶ sample points on the surface ─▶ ~600×6 model cloud
                                                                   (Data/ppf_library.npz)
```

All steps run **in one process** (`fp_server.py`). The mask, the depth and the
camera matrix are already in memory after step 1, so moving PPF into a separate
service would only add a network hop.

---

## Step 0: What arrives at the server

The laptop **does not send a point cloud. It sends images.** The server builds the
point cloud itself (step 2).

| file | contents | example (Data/Input) |
|---|---|---|
| `rgb.png` | colour image, 8 bit, 3 channels | 1280×720 |
| `depth.png` | depth image, 16 bit, one channel. Each pixel holds **Z (depth along the optical axis), not the ray distance**, in raw sensor units | 1280×720, mm |
| `camera.json` | `cam_K`: the 3×3 intrinsic matrix (9 numbers); `depth_scale`: raw units → mm | `fx=919.5, fy=918.9, cx=650.9, cy=350.6`, `depth_scale=1.0` |
| `click` (optional) | `{"u":640,"v":360}` or a list of points + labels | |

The server then (`_receive_frame`, `read_depth`):

1. **Converts depth to metres:** `Z[m] = raw × depth_scale / 1000`.
2. **Zeroes invalid values:** below 1 mm (the sensor got no reading) or beyond
   `ZFAR` = 3 m.
3. **Requires RGB and depth to be the same size.** Depth must be *aligned to
   colour*, so that `rgb[v,u]` and `depth[v,u]` look at the same point. If the sizes
   differ the request is rejected; the server does not resample.

**Output:** `rgb` (720×1280×3), `depth` (720×1280, float32, metres), `K` (3×3).

---

## Step 1: SAM2 separates the object (mask)

**Input:** `rgb` and one or more clicks. Depth is **not used** in this step.

**What SAM2 is and is not:**

- It is **not** panoptic segmentation. It does not label everything in the scene
  and knows no class names.
- It is **not** region growing. It does not grow pixel by pixel.
- It is a **promptable** network: it answers "which object contains this point?"
  with a mask. It has three parts:
  1. an *image encoder* (Hiera) that turns the image into a feature map once,
  2. a *prompt encoder* that encodes the clicks (left = object, right = not object),
  3. a *mask decoder* that combines both into a mask plus a predicted quality
     score (predicted IoU).
- Model used: `sam2.1_hiera_small`.

**How is the ground plane kept out?** There is no separate ground-plane removal.
The click is on the part, so SAM2 returns the region containing that point that is
coherent in colour and texture. The table is a different region and stays out. If
it leaks in, a right click (negative point) carves it away.

**A single click is ambiguous.** One point can mean "the whole part", "one face of
it" or "the hole in it". So for a single click SAM2 is asked for 3 candidate masks
and the one with the highest predicted score is kept. With several clicks, a single
mask is requested.

**Where the click comes from** (in order of preference):

1. if the client uploaded a `mask` file, it is used as is (SAM2 is skipped),
2. if the client sent a `click` field, SAM2 runs on those points,
3. otherwise a window opens on the server, an operator clicks the object and
   presses ENTER.

**Output:** `mask`, 720×1280 boolean (object = 1, everything else = 0).

---

## Step 2: Mask + depth → point cloud

This is where the explanation broke down in the meeting. The code does exactly the
following (`scene_cloud_from_mask` in `ppf_classifier.py`):

### 2a. Multiply mask and depth (Hadamard product)

The mask and the depth have the same size. Multiplying element by element sets
every pixel outside the mask to 0 and keeps the real Z inside it. This is still a
720×1280 **image**, not a point cloud yet.

### 2b. Back-projection: pixel → 3D point

For every mask pixel `(u, v)` with a valid Z (> 1 mm) the camera model is run in
reverse:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z = depth[v, u]
```

`fx, fy, cx, cy` come from `K` in `camera.json`. The result is an **N×3 matrix**:
N points, each (X, Y, Z) in metres in the camera frame. This is the "3×N" from the
meeting with rows and columns swapped. This cloud and everything after it is in the
**camera frame**, not the robot base frame.

### 2c. Cleaning and normals

PPF is built entirely on surface normals (step 3), so this part matters:

| operation | why |
|---|---|
| The mask is **eroded** by 2 px | Pixels on the silhouette straddle the object and the background. Their depth is a mix of both and forms a skirt of geometry that does not exist. |
| Normals are computed on a slightly **dilated** region (4 px), then the border is dropped | A point on the border has neighbours on one side only, so its normal tips inward. Computing on a wider region and then dropping the border gives every kept point a full neighbourhood. |
| Normal estimation: a plane is fitted to each point's 12 nearest neighbours (OpenCV `computeNormalsPC3d`) | |
| Normals are flipped **toward the camera** | CAD face normals point outward. Scene normals must point outward too (toward the camera), or the PPF angles do not line up. |
| Voxel downsampling, voxel = max(object diameter / 60, 1.5 mm), capped at **2000 points** | 2000 points is plenty for one object. More only costs time. |

**Output:** the scene cloud, **N×6** (`x, y, z, nx, ny, nz`), N ≤ 2000, metres.
Because it is seen from one side, this cloud is **partial**: only the faces looking
at the camera are there, not the bottom or the back (an *occluded view*).

---

## Preparation (offline): CAD → model cloud

For PPF to compare anything, every CAD model also needs a point cloud. This is done
once, when a part is added to the library, not in the live path
(`build_ppf_library.py`; the server also does it at startup if the `.npz` is
missing).

**What is in a PLY file?** The `.ply` exported from FreeCAD is a **triangle mesh**:
vertices, plus which three vertices form each triangle (faces). It is not a point
cloud. For example `test_objv2_ear.ply` has **8 vertices and 12 triangles**: a
plain box. Using the vertices directly as a cloud would give 8 points and nothing to
match. So:

1. The mesh is loaded and scaled to metres with `MESH_SCALE` (0.001, mm → m).
2. **60 000 points are sampled uniformly by area** on the surface
   (`trimesh.sample_surface`). Big triangles get many points, small ones few. Each
   point takes the **exact normal** of the triangle it lies on.
3. The cloud is thinned with a voxel grid to **~600 points**. One representative
   point is kept per voxel, not an average. Averaging would round off the box's
   corners and create 45° normals between two faces that belong to neither.
4. Saved: the model cloud (**~600×6**), its diameter, and its axis-aligned extents.

**Does the CAD have to be watertight?** Sampling is by surface area, so the mesh
does not strictly need to be closed. But the normals **must point outward**
(because scene normals are flipped toward the camera, i.e. outward). Meshes that
FreeCAD exports from solids already satisfy this.

> **Correction to the meeting:** I said "2500 points". The real values in the code
> are **~600** per model and **at most 2000** for the scene. They do not need to be
> equal (step 3 explains why).

---

## Step 3: PPF answers "which CAD is this?"

**Input:** the scene cloud (N×6) and every library model's cloud (~600×6).
**Output:** a **name** only (`"test_objv2_ear"`), plus the score table for all
models.

The library is built on OpenCV's `cv2.ppf_match_3d` module (Drost et al., 2010),
with one trained detector per model.

### 3a. Point Pair Feature: 4 numbers describe **a pair of points**

In the meeting this came across as "a 4-number row per point, a 2500×4 feature
matrix". **That is not what it is.** The feature does not describe one point; it
describes **the relationship between two points**. With points `m1`, `m2`, normals
`n1`, `n2` and `d = m2 − m1`:

```
F(m1, m2) = ( ‖d‖ ,     ∠(n1, d) ,  ∠(n2, d) ,  ∠(n1, n2) )
              distance  angle       angle       angle between the normals
```

These four numbers **do not change** however the object is placed or rotated:
distances and angles are invariant under rigid motion. That is why PPF can compare
without knowing the pose.

### 3b. Training (per model): a hash table

1. The model cloud is thinned again with voxels of 4 % of the model's diameter.
2. F is computed for **every pair of points**.
3. F is quantized: distance into bins 4 % of the diameter wide, angles into 30
   bins (12° each).
4. The quantized F becomes a **key**, and the hash table records "this key occurs
   at these model point pairs".

So there is no matrix; there is a **dictionary from feature → model pairs**. OpenCV
cannot write this table to disk, so the server retrains every model at startup
(~1–2 s per model). The `.npz` holds the sampled model clouds, not the table.

### 3c. Matching: by voting (no ordering needed)

The right question in the meeting was: *"How does it know which point corresponds
to which? Stacking the matrices would assume both are in the same order."* Answer:
**it does not know; it finds out by voting.** Point order and point count do not
matter:

1. Every 5th scene point is taken as a **reference point** `s_r`.
2. `s_r` is paired with the other scene points, and `F(s_r, s_i)` is computed for
   each pair.
3. That F is looked up in the hash table. Each model pair `(m_r, m_i)` found there
   says: "`s_r` may be model point `m_r`, and to line the pairs up you rotate by
   `α` about the normal."
4. That suggestion adds **one vote** to cell `[m_r, α]` of an accumulator kept for
   `s_r`.
5. The cell with the most votes is the most consistent correspondence. A **pose
   hypothesis** (4×4) is computed from it.

This is a Hough transform. Wrong matches scatter their votes over random cells;
correct matches keep landing in the same cell.

### 3d. Choosing the model: verification score, not vote count

Vote counts **cannot be compared between models**. A big model with many points and
self-similar surfaces collects more votes than a small one even when it is not the
object in the scene. So for each candidate model:

1. Take the **12 highest-voted pose hypotheses**.
2. Refine each with **50 iterations of ICP** (to remove the error left by PPF's 12°
   angle quantization).
3. Place the model in the scene at that pose and keep only the points **visible**
   from the camera (back-facing surfaces and self-occluded parts are dropped).
4. Compute a two-sided score with `τ = 8 mm` (sensor noise):

```
coverage  = (scene points within τ of the model) / (scene points)
explained = (visible model points within τ of a scene point) / (visible model points)
score     = coverage × explained          (between 0 and 1)
```

With `coverage` alone, a big model wins by blanketing the scene. With `explained`
alone, a small model wins by hiding inside it. The product penalizes both.

5. A model's score is the score of its best hypothesis. **The highest score wins.**
   The gap between first and second (the **margin**) says how sure we are.

### 3e. Size pre-filter (before matching)

Matching is expensive, so a cheap elimination runs first. The scene cloud's
diameter (99.5th percentile of pairwise distances) is compared to each model's:

- scene **more than 15 % larger** than the model → rejected (a partial view cannot
  be bigger than the real object),
- scene **smaller than 45 %** of the model → rejected (occlusion can shrink what is
  visible, so this side is kept loose).

On the real test frame this removed 8 of 13 models.

### 3f. Why the PPF pose is thrown away

PPF computes a pose in 3c–3d. But that pose only exists to compute the score (to
ask whether a CAD explains a cloud you have to put it somewhere). It **never leaves
the classifier.** FoundationPose computes the pose from scratch. Whether this is
the right choice **has not been measured yet** (see
[Action items](#action-items-from-the-meeting)).

---

## Step 4: FoundationPose computes the pose

**Input:** `K`, `rgb` (720×1280×3), `depth` (metres), `mask`, and the `.ply` chosen
in step 3 (scaled to metres).

**Output:** `T` (4×4), **the object's pose in the camera frame**: the top-left 3×3
is the rotation, the right column the translation (metres). The server does not
transform it into the robot base frame.

What happens inside `register()` (`estimater.py`):

1. Depth is eroded and smoothed with a bilateral filter (noise reduction).
2. **Translation guess:** the centre pixel of the mask's bounding box is
   back-projected at the median depth inside the mask. Every hypothesis starts from
   this point.
3. **Rotation hypotheses:** 42 viewing directions on a sphere around the object
   (an icosphere) × 6 in-plane rotations each (every 60°) = 252 candidates.
   Candidates closer than 30° to each other, or identical under symmetry, are merged.
4. **Refiner network:** for each hypothesis the CAD is rendered at that pose,
   compared with the same region of the real RGB-D, and the network predicts a
   correction. 5 iterations (`EST_REFINE_ITER`).
5. **Scorer network:** ranks all refined hypotheses; the best one is returned.

Notes:

- Our CAD models have no texture, so the render is flat gray. Matching therefore
  relies mostly on geometry and depth.
- The returned `score` (67.97 in the last run) is an unnormalized ranking score,
  not a probability. PPF's `classification.score` is 0–1.
- The networks are object-agnostic and loaded once at startup. Switching CAD is a
  `reset_object()` call that takes milliseconds.
- FoundationPose also has a **tracking mode** (`track_one`). `/predict_pose` does not
  use it; it runs on the laptop for live tracking, starting from this pose. See
  [Live tracking](#live-tracking-foundationpose-on-the-laptop).
- `REGISTER_CROP=1` registers on the square around the mask instead of the whole
  frame (off by default). Only for GPUs with 8 GB: see the notes in Live tracking.

**Example from the last run** (`Data/Output/foundationpose_results/detection_pem.json`):
object `test_objv2_ear`, translation ≈ (164, 45, 499) mm, i.e. about 50 cm from the
camera, to the right and below the optical centre.

---

## What comes back

```jsonc
{
  "status": "success",
  "object_name": "test_objv2_ear",        // the .ply the ICP node must load
  "object_file": "test_objv2_ear.ply",
  "pose": [[…4x4…]], "units": "m",        // FoundationPose's pose, not PPF's
  "score": 67.97,                         // FoundationPose ranking score
  "classification": {
    "score": 0.588, "margin": 0.438,      // PPF verification score (0–1) and gap
    "runner_up": "test_objv2_base",
    "scores":   { "test_objv2_ear": 0.588, "test_objv2_base": 0.150, … },
    "rejected": { "Tblock": "scene 255mm exceeds CAD 181mm by 41%", … },
    "elapsed_sec": 1.4
  },
  "artifacts": { "detection_pem.json", "detection_ism.npz", "mask.png",
                 "vis_pose.png", "object_name.txt" }   // base64
}
```

`vis_pose.png` is the first thing to open when something looks wrong. If the box is
the **wrong shape**, classification was wrong (check `classification.scores`). If it
is the **right shape in the wrong place**, classification was right and the pose
was wrong.

`detection_pem.json` is in the old SAM-6D format and its translation is in
**millimetres** (so the ICP node can keep reading it unchanged). The `pose` field in
the HTTP reply is in **metres**.

The name reaches downstream three ways: `Detection3D.id` / `class_id`, the latched
`/perception/object_name` topic, and `object_name.txt` in the results directory.

---

## Live tracking: FoundationPose on the laptop

*Added 6 October 2026. Design and reasoning: [realtime_fp.md](realtime_fp.md).*

Registration (steps 0–4) answers "where is the part" **once**, in 1–4 s. To follow
a part while it is being placed, FoundationPose's second mode, `track_one`, is run
on **every camera frame**: it starts from the previous pose and only corrects it with
2 passes of the refiner network (no hypotheses, no scorer), so it takes tens of
milliseconds instead of seconds.

### Who does what

Registration stays on the desktop. Tracking runs **on the laptop's own GPU**
(RTX 4060), next to the camera, so camera frames never cross the network. Only one
frame per registration goes to the desktop.

```
 LAPTOP (camera, ROS 2 Jazzy, RTX 4060)                       DESKTOP (RTX 5070 Ti)

 camera ─▶ foundationpose_bridge_node ── HTTP, 1 frame ─────▶ fp_server.py :5000
                                       ◀─ pose + object name ─  /predict_pose (steps 0–4)
              │ /perception/detections (latched) = the seed
              ▼
 camera ─▶ fp_tracker_node  (ROS node, laptop host)
  30 Hz       │  ▲
              │  │  ws://127.0.0.1:5001, same machine, raw bytes
              ▼  │
          fp_track_server.py  (laptop container fp-sam2:ada)
          refiner network only, track_one per frame
              │
              ▼
   /perception/fp/pose (PoseStamped)  +  TF camera_color_optical_frame → fp_object
   (same message types as the ICP node)
```

| piece | where | file |
|---|---|---|
| registration server | desktop container | `scripts/fp_server.py` (unchanged role) |
| tracking-only server | laptop container | `scripts/fp_track_server.py` |
| tracking state machine + WebSocket server | both servers import it | `scripts/fp_tracking.py` |
| wire protocol (no torch) | everywhere | `scripts/fp_stream.py` |
| ROS client | laptop host (welding_cell_ws) | `scripts/fp_tracker_node.py` |
| test client without ROS | any host | `scripts/track_replay.py` |
| GPU benchmark | container | `scripts/track_bench.py` |

Why a WebSocket between the ROS node and the tracker, on the same laptop: the
container is Ubuntu 22.04 / Python 3.10 without ROS, the workspace is ROS 2 Jazzy /
Python 3.12, and Humble↔Jazzy DDS is not supported. On localhost the socket only
costs a few milliseconds of copying.

### One frame, step by step

**Seed.** The bridge publishes the registered pose on `/perception/detections`
(latched). The node sends it to the tracker as `start`: the pose `T` (4×4, camera
frame, metres), the object name (selects the `.ply`), and `T_base_cam` from TF at the
detection's stamp.

**Each frame** (the node takes the newest RGB-D pair; at most 2 frames are in flight,
older ones are skipped, never queued):

| | in | done | out |
|---|---|---|---|
| node | `/camera/color/image_raw` (rgb8), `/camera/depth/image_rect_raw` (16UC1, mm), `camera_info`, TF | pair by stamp, look up `T_base_cam` at the image stamp, pack header + raw bytes | one binary message, ~4.6 MB at 1280×720 |
| tracker | header (seq, stamp, K, depth scale, `T_base_cam`), RGB, depth | depth → metres; **ego-motion prediction** (below); `track_one`, 2 refiner passes; **fit** check (below) | JSON: pose 4×4 or null, state, fit |
| node | reply | publish if TRACKING | `PoseStamped` + TF, stamped with the **image** stamp |

**Ego-motion prediction.** The camera is on the robot's wrist. When the arm moves,
a still part jumps in the image. Before tracking, the tracker moves the previous pose
by the camera's own motion:

```
pose_predicted = inv(T_base_cam_now) · T_base_cam_previous · pose_previous
```

so `track_one` only has to follow the part's own motion. Without TF this step is
skipped and arm motion looks like part motion.

**Fit: is the pose still right?** `track_one` returns a pose but no score. So after
each frame the CAD is rendered at the new pose and compared with the measured depth:

```
fit = (rendered CAD pixels whose measured depth is within 10 mm) / (rendered CAD pixels)
```

**States:**

```
IDLE ──start──▶ TRACKING ──fit < 0.5 for 3 frames──▶ LOST ──fit ≥ 0.6──▶ TRACKING
```

While LOST:
1. no pose is published (the TF stops; nothing wrong is shown),
2. every frame the tracker retries cheaply from the last good pose (a hand that
   briefly covered the part),
3. every 3 s the node re-registers through the desktop **without a click**: it asks
   the tracker for the CAD's silhouette at the last good pose (dilated), sends that
   frame + mask to `/predict_pose`, and restarts tracking from the answer.

### Measured (RTX 4060 laptop)

`track_bench.py`, GPU only, saved 1280×720 frame, still part, 200 frames each:

| refine passes | input | ms/frame mean (p95) | rate | jitter |
|---|---|---|---|---|
| 1 | full frame | 28.3 (40.1) | 35 Hz | 0.5 mm, 0.16° |
| 2 | full frame | 43.2 (59.9) | 23 Hz | 0.5 mm, 0.19° |
| 1 | ROI crop 524×478 | 23.6 (34.8) | 42 Hz | 0.7 mm, 0.22° |
| 2 | ROI crop | 38.8 (56.1) | 26 Hz | 0.7 mm, 0.27° |

The tracking-only server uses ~250 MB of GPU memory.

`fp_tracker_node` end to end, with a test publisher in place of the camera (saved
frame at 30 Hz) and both servers on the same 4060:

| | result | target |
|---|---|---|
| `/perception/fp/pose` rate | 25–29 Hz | ≥ 15 Hz (goal 30) |
| image stamp → pose | 70–90 ms (raw bytes); ~100 ms with JPEG/PNG | ≤ 100 ms (goal 50) |
| image blanked for 1.5 s | LOST after 3 frames, recovered by itself (fit 0.90) | |
| re-registration through `/predict_pose` | 3.9–4.0 s, tracking restarted | |

During a registration on the same GPU tracking dropped to ~8 Hz. In the real split
the desktop registers on its own GPU, so this does not happen.

### Notes and limits

- **Registration does not fit on the laptop.** A full-frame `register()` needs more
  than 8 GB: the scorer warps all 252 hypotheses to full-frame size at once (one
  2.6 GB allocation). `REGISTER_CROP=1` crops the frame to the region the networks
  look at and fits (peak 4.3 GB), but it is **not verified** to give the same pose,
  so it is off by default and only used for single-machine tests.
- **The fit check is fooled by flat parts on flat surfaces.** A thin plate lying on
  a table fits the depth in any in-plane position. On the test frame a wrong pose
  scored 0.94.
- **The test frame has no real part in it.** Its mask is a strip on the flat plate.
  Rate, delay and the LOST logic are measured; **pose accuracy of tracking is not**.
  Next: record a sequence with a real part moving.
- **The seed's stamp.** The bridge stamps `/perception/detections` with the publish
  time, not the captured frame's stamp. If the arm moves between capture and reply,
  the seed is off by that motion (tracking pulls in small offsets; large ones end in
  LOST and a re-seed). Fix in the bridge: `out.header.stamp = frozen.stamp`.
- **Full frame by default.** The node can crop to the region around the part
  (`use_roi`), which cuts the copy and the depth filtering, but with the tracker on
  localhost there is little to gain.
- ICP is unchanged and still does the stationary steps (`save_object`,
  `refine_pose`). Handing tracking over to `/perception/fp/pose` in the ICP node
  (`tracking_source: fp`) is the next step (realtime_fp.md, step 6).

---

## Answers to the meeting questions

**How does SAM2 know which object? Is it configured somehow?**
It does not know; we tell it, with a click (a pixel coordinate) on the object. SAM2
returns the mask of the region containing that point. No class names, no training,
no configuration. See step 1.

**Is segmentation done on RGB or on depth?**
RGB only. Depth comes in after the mask exists.

**Is the ground plane removed?**
Not as a separate step. The mask already leaves it out. Mixed pixels on the border
are dropped by erosion.

**Is a point cloud sent to the server?**
No. The laptop sends `rgb.png`, `depth.png` (16 bit, mm) and `camera.json` (K).
Masking and conversion to a point cloud happen on the server, which is why the
camera model (K) is sent along.

**How does the depth image become a point cloud?**
It is multiplied element-wise by the mask, then every mask pixel is back-projected
with K: `X=(u−cx)Z/fx, Y=(v−cy)Z/fy`. The result is N×3 (N×6 with normals). See
step 2.

**Where does the CAD point cloud come from? Isn't a PLY already a point cloud?**
No. A PLY is a triangle mesh, and our parts have only a handful of vertices (a box
has 8). 60 000 points are sampled on the surface by area and thinned to ~600; each
point takes the normal of its triangle. See Preparation.

**What is the "2500×4 feature matrix"?**
There is no such matrix; I explained it wrongly in the meeting. The 4 numbers
(distance + 3 angles) describe **a pair of points**, not one point, and they are
stored as hash-table keys, not as a matrix. See 3a–3b.

**How can two clouds with different point counts and orderings be compared?**
By voting. Points are never stacked in order. Each scene pair looks up similar
model pairs in the table and votes; the consistent correspondence collects the
votes. Point counts do not need to match. See 3c.

**Is there ICP inside? If so, why do we need PPF?**
Yes, but in a small role. ICP does not work without a good starting pose. PPF's
voting provides that start, and ICP only corrects it by a few millimetres (3d,
item 2). The score is computed on the corrected pose.

**Is PPF scale-invariant? Would it confuse a 13 cm and a 25 cm part of the same shape?**
No, it is not scale-invariant, and that is the behaviour we want:
- A depth camera measures metrically. Looking from farther away does **not** make
  the cloud smaller; it only gives fewer points and more noise. So there is no
  scale ambiguity to solve.
- The `‖d‖` in the feature is in metres. Pairs on a part twice the size fall into
  different hash bins.
- Before that, the size pre-filter (3e) never even matches a 25 cm scene against a
  13 cm model.

**What if the distinguishing feature is hidden underneath (e.g. an emboss on the bottom face)?**
It cannot be distinguished from one view. The two models get close scores and the
`margin` shrinks. That is not a bug; it is an honest signal that the information is
not there. The fix is to look from another angle (see Action items). Similarly,
`plate` (150×4×100 mm) and `test_objv1_base` (180×4×100 mm) are very hard to tell
apart from one view.

**If PPF also gives a pose, why use FoundationPose?**
There is **no measured justification yet.** The PPF pose is computed inside the
classifier and discarded. FoundationPose was assumed to be more accurate but this
was not tested. The comparison is on the action list.

**Which frame are the poses in?**
All in the camera frame (`camera_color_optical_frame`). Conversion to the robot
base happens downstream, not on the server.

**How long does it take?**
- Classification (steps 2 + 3): ~1.4 s with a 13-model library; reported in the
  reply as `elapsed_sec`.
- Detector training at startup: ~1–2 s per model.
- The whole chain (SAM2 + PPF + FoundationPose): under 10 s by observation, **not
  measured step by step.** The old SAM-6D pipeline took 1–1.5 minutes.
- Building the library clouds is offline and does not count toward live latency.

---

## Action items from the meeting

Perception:

- [ ] **Compare the PPF pose with the FoundationPose pose.** Accuracy and time on
      3–5 different scenes. The case for FoundationPose should rest on this table.
      The PPF pose is already computed; it just needs to be returned.
- [ ] **Compare FoundationPose tracking (`track_one`) with ICP tracking.** The
      tracking path now exists ([Live tracking](#live-tracking-foundationpose-on-the-laptop));
      run `pose_jitter_probe.py` on `/perception/fp/pose` and
      `/perception/icp/refined_pose` with a still and a moving part.
- [ ] **Measure timings** separately: PPF matching per model, SAM2, FoundationPose
      `register()`.
- [ ] **Find the camera's best working distance.** How depth error changes with
      distance (the sensor's minimum range is ~28 cm; where is accuracy best?).
      `PPF_TAU` (8 mm) should be set from this measurement.
- [ ] **Two-stage viewing:** identify from far (which CAD?), then move to the best
      distance and re-estimate the pose. Recognition and precise pose are two
      separate jobs.
- [ ] **Re-estimate the pose once more after the part is placed**, to reset the few
      millimetres of error accumulated during tracking.
- [ ] **On ambiguous classification, look from another angle.** When the `margin`
      is small, pick the viewpoint that best separates the candidates
      (*next-best-view*).
- [ ] Check that CAD normals point outward and that export units are correct.

Out of scope for this document (weld-seam side, to be written separately): finding
the contact boundary (seam) by letting two points on the CAD models "walk" toward
each other, its step-by-step visualisation and a proof of why the walk stops at
edges; marking targets with a laser pointer; the journal target by the new year.

---

## Running it

```bash
docker/run_container_blackwell.sh          # X11 forwarded, workspace bind-mounted

# once per new part (fp_server does this automatically if the .npz is missing)
python scripts/build_ppf_library.py        # Data/Input/*.ply -> Data/ppf_library.npz
python scripts/ppf_selftest.py --views 3   # test the library on synthetic views

python scripts/fp_server.py                # listens on :5000
```

Startup loads the FoundationPose networks, then the PPF detectors (~1–2 s per
model), then SAM2.

**Live tracking (laptop).** Build the laptop image once (Ada, sm_89), then:

```bash
docker build -f docker/Dockerfile.blackwell --build-arg TORCH_CUDA_ARCH_LIST=8.9 -t fp-sam2:ada .
docker/run_container.sh                    # laptop container
python scripts/fp_track_server.py          # inside it: listens on 127.0.0.1:5001

# laptop host, ROS 2
ros2 run admittance_control fp_tracker_node.py --ros-args \
    -p register_url:=http://<desktop>:5000/predict_pose

# without ROS: stream a saved frame, simulate motion / a covered part / a re-seed
python3 scripts/track_replay.py --register-url http://<desktop>:5000/predict_pose \
    --wobble 40 --blackout 5 1.5 --force-reseed-at 10
python scripts/track_bench.py              # GPU-only timing (inside the container)
```

**Use `/classify` during bring-up.** It stops after step 3, spends no GPU time on a
pose, and returns the whole score table:

```bash
curl -F rgb=@rgb.png -F depth=@depth.png -F camera=@camera.json \
     -F mask=@mask.png  http://localhost:5000/classify
```

Watch the `margin` between first and second place across viewpoints. Once you know
what real margins look like, set `PPF_MIN_MARGIN` and `PPF_STRICT=1` so the server
refuses (HTTP 422) instead of guessing.

**Adding a CAD model without restarting:**

```bash
curl -F model=@new_part.ply http://localhost:5000/add_model
# or from ROS:
ros2 param set /foundationpose_bridge model_ply_path /path/to/new_part.ply
ros2 service call /foundationpose_bridge/add_model std_srvs/srv/Trigger
```

Each model has its own detector, so adding one retrains nothing else. The reply
lists existing models within 10 % of the new one's diameter. The size filter cannot
separate those; they compete on score alone.

### Endpoints

| endpoint | purpose |
|---|---|
| `POST /classify` | steps 0–3. For bring-up. |
| `POST /predict_pose` | steps 0–4, the whole chain |
| `POST /add_model` | index a new `.ply` into the live library and persist it |
| `GET /health` | what is loaded, what is in the library, tracking state |
| `ws://:5001` | live tracking endpoint of `fp_server.py` itself (`TRACK_ENABLE`). Not used in the laptop/desktop split; `fp_track_server.py` serves the same protocol on the laptop |

### Configuration (environment variables, all optional)

| var | default | meaning |
|---|---|---|
| `MESH_SCALE` | `0.001` | CAD units → metres, applied to **every** model |
| `CAD_DIR` | `Data/Input` | directory scanned for library `.ply` files |
| `PPF_ENABLE` | `1` | `0`: classification off, always use `MESH_PATH` |
| `PPF_TAU` | `0.008` | verification radius in metres, i.e. sensor noise |
| `PPF_MIN_MARGIN` / `PPF_STRICT` | `0.0` / `0` | ambiguity threshold / refuse when ambiguous |
| `EST_REFINE_ITER` | `5` | FoundationPose refinement iterations |
| `ZFAR` | `3.0` | depth beyond this many metres is discarded |
| `SYMMETRY_INFO` | – | BOP-style symmetry info for symmetric parts |
| `TIMING_CSV` | `Data/Output/timings.csv` | where per-stage timings are appended |
| `REGISTER_CROP` | `0` | `1`: register on the region around the mask (8 GB GPUs; unverified) |
| `TRACK_ENABLE` / `TRACK_PORT` | `1` / `5001` | `fp_server.py`'s own tracking endpoint |

`fp_track_server.py` (laptop) reads `CAD_DIR`, `MESH_SCALE`, `MESH_PATH`, `ZFAR`,
`TRACK_HOST` (`127.0.0.1`) and `TRACK_PORT` (`5001`).

`fp_tracker_node` parameters (defaults):

| parameter | default | meaning |
|---|---|---|
| `track_url` | `ws://127.0.0.1:5001` | the tracking-only server |
| `register_url` | `http://127.0.0.1:5000/predict_pose` | the desktop, for re-registration |
| `rgb_topic` / `depth_topic` / `camera_info_topic` | `/camera/color/image_raw` / `/camera/depth/image_rect_raw` / `/camera/color/camera_info` | camera input, paired by stamp |
| `detections_topic` | `/perception/detections` | the seed (from the bridge) |
| `pose_topic` / `status_topic` / `marker_topic` | `/perception/fp/pose` / `/perception/fp/status` / `/perception/fp/marker` | outputs |
| `base_frame` / `child_frame` | `base_link` / `fp_object` | ego-motion TF / published TF |
| `auto_start`, `max_seed_age_sec` | `true`, `30` | start on every new detection younger than this |
| `max_in_flight` | `2` | frames sent but not yet answered |
| `refine_iter` | `2` | refiner passes per frame (1 is faster, more jitter) |
| `rgb_encoding` / `depth_encoding` | `raw` / `raw` | `jpeg` / `png` only if the tracker is on another machine |
| `use_roi`, `roi_margin_px` | `false`, `40` | crop around the part |
| `ego_motion` | `true` | cancel arm motion with TF |
| `fit_tol_m`, `fit_lost`, `fit_recover`, `lost_frames` | `0.010`, `0.5`, `0.6`, `3` | LOST detection |
| `reseed`, `reseed_interval_sec` | `true`, `3.0` | re-register through the desktop while LOST |
| `mesh_resource`, `mesh_scale` | `package://admittance_control/models/{object}.ply`, `0.001` | RViz mesh marker |

Services: `~/start` (track from the last detection), `~/stop`, `~/reseed`.

Full list and PPF details: [docs/PPF.md](docs/PPF.md).

### Timing

Every request (and the server startup) is timed per stage. The same numbers go to
three places:

- **terminal**: a table printed after each request,
- **`Data/Output/timings.csv`**: appended, one row per stage and one row per CAD for
  PPF (columns `timestamp, request_id, endpoint, status, object_name, stage, cad,
  seconds, detail`). It is long format: pivot on `stage` / `cad` in a spreadsheet,
- **the reply**: a `timings` field with the stages, `total_sec` and `compute_sec`.

Example (`/predict_pose`, warm server, RTX 5070 Ti, 14 CADs in the library):

```
── timing /predict_pose #3  object=test_objv2_ear  [success] ──────────────
  receive + decode frame                    0.025 s
  segmentation (SAM2)                       0.053 s
  mask -> scene point cloud                 0.155 s   2000 pts
  ppf: scene prep + extent filter           0.050 s
      test_objv2_ear                        0.195 s   match=0.106s verify=0.089s score=0.764
      test_objv2_base                       0.244 s   match=0.154s verify=0.090s score=0.210
      ...
      270circle                             0.000 s   rejected by extent filter: ...
  CAD load / switch                         0.000 s
  FoundationPose register                   1.198 s   5 refine iters
  overlay + artifacts                       0.095 s
  TOTAL                                     4.506 s
```

Notes for reading the numbers:

- **The first request after startup is slower** (CUDA warm-up). Leave it out of
  averages.
- **Per-CAD PPF time** = `match` (OpenCV voting) + `verify` (ICP polish +
  coverage·explained score). A CAD rejected by the size pre-filter costs 0 s.
- **Interactive window:** the time the operator spends clicking is recorded as its
  own stage (`operator clicking`) and excluded from `TOTAL excl. operator` /
  `compute_sec`. `segmentation (SAM2)` is the network time for the last click only.
- **`TOTAL` is wall-clock time** from the start of the request. It also includes small
  untimed gaps, such as the depth check and the PPF result bookkeeping.
- **Startup** rows show the one-off model loading and the PPF training time for each
  CAD. Training runs inside `load PPF library`.

---

## CAD units: read this before adding a model

`MESH_SCALE` is applied identically to **every** model in the library. FreeCAD's
export unit, however, is a **per-document** setting. A single part exported in
metres or centimetres enters a millimetre library at 1000× or 10× the wrong size.
Because the size filter compares physical size, such a model becomes **silently
unclassifiable**.

| check | catches | misses |
|---|---|---|
| absolute plausibility (10 mm – 1.5 m after scaling) | the 1000× case, and suggests the right scale | – |
| >5× off the library median | one drifted file among correct ones | a library that is genuinely that diverse |
| **the extents table `build_ppf_library.py` prints** | everything | nothing, but you have to look |

A 10× slip on an isolated part cannot be caught automatically (a 25 mm part looks
reasonable on its own). You know what your parts measure; a glance at that table
is the only reliable check.

---

## Failure modes

| symptom | likely cause |
|---|---|
| size filter rejected every model | bad depth, or the mask leaked onto the background. The filter is then ignored, so the table is still readable. |
| the right part scores near zero against itself (self-test) | that model may need a finer sampling step. `ppf_selftest.py` prints the `--add --sampling-step` command; re-run the self-test to confirm it helped, because it often does not. On the curved parts (`C1`, `R2`, `RR1`, `E2`, `S3`, `SP3`) the real cause was ICP polish sliding correct poses away, now fixed by also scoring the unpolished pose. |
| `assembly_mesh` loses to `test_objv2_base` / `test_objv2_ear` | an assembly contains its parts' surfaces, so a view showing mostly one part is explained by that part alone. Expected from one view. |
| two parts keep swapping, small margin | they really are similar from one view (`plate` ↔ `vplate`). Raise `PPF_MIN_MARGIN`. |
| one model never wins anywhere | check its printed extents; almost always an export-unit slip |
| box in `vis_pose.png` is the right shape in the wrong place | classification right, pose wrong |
| box in `vis_pose.png` is the wrong shape | classification wrong; check `classification.scores` |

## Known limits

- Two thin plates of similar size cannot be separated from one view. That is a
  property of the parts; the `margin` is the signal.
- Thin curved parts need a finer sampling step than the 0.04 default.
- `test_objv2` and `test_objv3` have identical bounding boxes, so the size filter
  cannot help. They separate cleanly on score (~0.8 vs ~0.05).
- The self-test is synthetic and uses exact CAD normals; real depth is noisier.
  Read self-test accuracy as an upper bound.
- Evaluation on real data currently rests on **a single frame** (`test_objv2_ear`,
  margin 0.438). That is not an evaluation.
- Live tracking: see [Notes and limits](#notes-and-limits) (no registration on the
  8 GB laptop, the fit check on flat parts, tracking accuracy not yet measured).
