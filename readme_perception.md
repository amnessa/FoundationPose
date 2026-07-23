# Host-side perception: SAM2 → PPF → FoundationPose

This is the half of the system that runs on the desktop with the GPU. The laptop
client never sees any of it: it sends an RGB-D frame and gets back an object name
and a pose.

```
  LAPTOP / other container                    HOST PC (this repo, in Docker)
  ────────────────────────                    ──────────────────────────────
  foundationpose_bridge_node
    ros2 service call ~/trigger
        │
        │  POST rgb.png + depth.png + camera.json
        ▼
                                        fp_server.py  :5000
                                              │
                                    ┌─────────┴──────────┐
                                    │  1. SAM2           │  click or uploaded mask
                                    │     → binary mask  │
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  2. mask ∧ depth   │  → oriented point cloud
                                    │     → point cloud  │    (camera frame, metres)
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  3. PPF classifier │  vs. the CAD library
                                    │     → "objv2_ear"  │    a NAME, only
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │  4. FoundationPose │  register() on that .ply
                                    │     → 4x4 pose     │
                                    └─────────┬──────────┘
        │                                     │
        │  ◀── {object_name, pose (metres), classification, artifacts}
        ▼
  publishes Detection3DArray + /perception/object_name
        │
        ▼
  laptop ICP node: loads that .ply, snaps to the pose, tracks at 60 Hz
```

The three stages run **in one process**. There is no PPF "node" and no second
service: the mask, the depth and the camera matrix are all already in
`fp_server.py`'s memory after stage 1, and shipping a point cloud over HTTP to get
a string back would add a network hop for nothing. The classifier is still a
self-contained module (`scripts/ppf_classifier.py`) with its own HTTP endpoint
(`/classify`), so it can be split out later without touching its internals.

---

## Stage 1 — SAM2 gives the mask

FoundationPose has no detector. It must be *told* which pixels are the object.
Unchanged from before PPF existed, and in preference order:

1. a `mask` file uploaded with the request,
2. a `click` field (`{"u":…, "v":…}`) the client sends,
3. otherwise an OpenCV window opens **on the host**, and an operator clicks the
   object. Left click adds, right click subtracts, ENTER accepts.

The interactive path needs `DISPLAY`; `run_container_blackwell.sh` already forwards
X11. It also means a trigger blocks until someone clicks, which is why the bridge's
`request_timeout_sec` defaults to 300 s.

## Stage 2 — mask + depth → point cloud

`scene_cloud_from_mask()` in `ppf_classifier.py`. Three details matter more than the
projection arithmetic:

- **The mask is eroded first** (2 px). Depth pixels straddling the silhouette mix
  the object with the background, and those mixed points form a skirt of invented
  geometry that drags every verification score down — unevenly, since it hurts thin
  parts most.
- **Normals are fitted on a dilated crop, then the border is discarded.** A point on
  the mask boundary has neighbours on one side only, so its normal tips inward. PPF
  is built entirely out of normals, so this is not cosmetic.
- **Normals are flipped toward the camera**, matching the outward-facing face
  normals of the CAD. The PPF angle features are sign-sensitive; a flipped scene
  normal lands the pair in a different hash bucket entirely.

The result is voxel-downsampled to ~2000 points. That is deliberately small — it is
one segmented object, not a scene.

## Stage 3 — PPF says which CAD it is

Built on OpenCV's `cv2.ppf_match_3d` (the `surface_matching` module), one trained
`PPF3DDetector` per model.

**It returns a name and nothing else.** Pose estimation is FoundationPose's job. A
transform *is* computed inside the classifier — there is no way to ask "does this
CAD explain this cloud?" without first placing the CAD somewhere — but it exists
only to compute the score, and is dropped at the end of `_score_model`. Nothing
downstream ever sees it.

Order of operations, cheapest first:

1. **Extent pre-filter.** Reject any CAD whose size is incompatible with the
   segment. On the real test frame this eliminated 8 of 13 candidates for
   essentially no cost. The upper bound is tight (15 %) because "the scene is bigger
   than the CAD" is physically impossible for a partial view; the lower bound is
   loose (55 %) because occlusion legitimately shrinks what you can see.
2. **Match** the surviving models. Each returns ranked pose hypotheses with vote
   counts.
3. **ICP-polish** each hypothesis (OpenCV's `ICP.registerModelToScene`, seeded with
   the PPF pose — refining the *unposed* cloud is an easy and silent mistake).
4. **Score** each with `coverage × explained` and keep the best per model.

The score, not the vote count, decides. Vote counts scale with model point count
and surface self-similarity, so they are not comparable between two different CAD
models — they only rank hypotheses *within* one model, where the scale is shared.

Full detail, including the two-sided score definition: [docs/PPF.md](docs/PPF.md).

## Stage 4 — FoundationPose poses it

The classified name selects a `.ply`; `use_cad()` swaps it onto the single resident
`FoundationPose` instance via `reset_object()`. The scorer and refiner networks are
the expensive part of construction and they are object-agnostic, so swapping CAD
between requests costs milliseconds, not a reload. That is what makes it affordable
to let the classifier choose the mesh per request.

`register()` then estimates the pose from scratch on that mesh. It does not receive,
and is not influenced by, anything PPF computed.

---

## CAD units — read this before adding a model

`MESH_SCALE` (default `0.001`, i.e. mm → m) is applied uniformly to **every** model
in the library and to the mesh FoundationPose registers against. FreeCAD's export
unit is a **per-document setting**, so a single part exported in metres or
centimetres drops into an otherwise-millimetre library at 1000× or 10× the wrong
size.

This is no longer a cosmetic problem. The extent pre-filter compares *physical size*,
so a mis-scaled model is silently unclassifiable — it gets thrown out of every scene
it belongs in, or swallows scenes it has nothing to do with. The failure is very
hard to read backwards from a bad classification, so it is checked when the model is
indexed:

| check | catches | misses |
|---|---|---|
| absolute plausibility (10 mm – 1.5 m after scaling) | the 1000× case, and suggests the right scale | — |
| library-relative outlier (>5× off the median) | one drifted file among correct ones | a library that is genuinely that diverse |
| **the extents table `build_ppf_library.py` prints** | everything | nothing — but you have to look |

A 10× slip on an isolated part is **not** automatically detectable: a 25 mm object is
perfectly reasonable on its own. That is why the builder prints every model's
extents in millimetres. You know what your parts actually measure; a glance at that
table is the only reliable check.

`--mesh-scale` applies to the entire library, so one oddly-exported file must be
re-exported rather than worked around at build time.

---

## Running it

```bash
docker/run_container_blackwell.sh          # X11 forwarded, workspace bind-mounted

# once per new part (fp_server does this automatically if the .npz is missing)
python scripts/build_ppf_library.py        # scan Data/Input/*.ply -> Data/ppf_library.npz
python scripts/ppf_selftest.py --views 3   # sanity-check the library before trusting it

python scripts/fp_server.py                # listens on :5000
```

Startup loads the FoundationPose networks, trains one PPF detector per model
(~1–2 s each; the OpenCV hashtable cannot be serialized, so this happens every
boot), and loads SAM2.

### Bring-up loop

Use `/classify` rather than `/predict_pose`. It stops after stage 3 — about 1.4 s,
no GPU pose — and returns the entire score table:

```bash
curl -F rgb=@rgb.png -F depth=@depth.png -F camera=@camera.json \
     -F mask=@mask.png  http://localhost:5000/classify
```

Watch the **margin** between the winner and the runner-up across viewpoints. That is
the number that tells you whether a threshold is safe. When you know what real
margins look like, set `PPF_MIN_MARGIN`, and `PPF_STRICT=1` to make the server
refuse (HTTP 422) rather than guess.

### Adding a CAD model without restarting

The motivating case is a part composed from two existing models and exported as a
new mesh:

```bash
curl -F model=@new_part.ply http://localhost:5000/add_model
```

or from the bridge, using a parameter plus stock `Trigger` so no custom service type
and no interface rebuild is needed:

```bash
ros2 param set /foundationpose_bridge model_ply_path /path/to/new_part.ply
ros2 service call /foundationpose_bridge/add_model std_srvs/srv/Trigger
```

Each model owns its own detector, so adding one retrains nothing else. The reply
carries the scale warning (if any) and lists existing models within 10 % of the new
one's diameter — the extent pre-filter cannot separate those, so they will compete
on score alone from now on.

### Endpoints

| endpoint | purpose |
|---|---|
| `POST /classify` | stages 1–3 only. The bring-up endpoint. |
| `POST /predict_pose` | the whole pipeline |
| `POST /add_model` | index a new `.ply` into the live library and persist it |
| `GET /health` | what is loaded, and what is in the library |

### Configuration

All optional, all environment variables. See [docs/PPF.md](docs/PPF.md) for the
full table; the ones that matter most:

| var | default | meaning |
|---|---|---|
| `MESH_SCALE` | `0.001` | CAD units → metres, applied to every model |
| `CAD_DIR` | `Data/Input` | scanned for library `.ply` files |
| `PPF_ENABLE` | `1` | `0` pins the server to `MESH_PATH`, exactly as before PPF |
| `PPF_TAU` | `0.008` | verification inlier radius in metres — your depth noise |
| `PPF_MIN_MARGIN` / `PPF_STRICT` | `0.0` / `0` | ambiguity handling |

---

## What comes back

```jsonc
{
  "status": "success",
  "object_name": "test_objv2_ear",        // what the ICP client must load
  "object_file": "test_objv2_ear.ply",
  "pose": [[…4x4…]], "units": "m",        // FoundationPose's, not PPF's
  "score": 93.56,                         // FoundationPose's hypothesis score
  "classification": {
    "score": 0.588, "margin": 0.438,
    "runner_up": "test_objv2_base",
    "ambiguous": false,
    "scores": { "test_objv2_ear": 0.588, "test_objv2_base": 0.150, … },
    "rejected": { "Tblock": "scene 255mm exceeds CAD 181mm by 41%", … }
  },
  "artifacts": { "detection_pem.json": "…", "detection_ism.npz": "…",
                 "object_name.txt": "…", "mask.png": "…", "vis_pose.png": "…" }
}
```

The whole score table is always present, winner or not. The runner-up margin is the
number worth watching, and carrying it costs nothing.

`score` (~93) is FoundationPose's unnormalized hypothesis-ranking score, not a
probability — do not carry a SAM-6D `min_score` threshold over unchanged.
`classification.score` is the PPF verification score and *is* 0–1.

### How the name reaches the tracking node

Three ways, so a consumer can pick whichever it already speaks:

- `Detection3D.id` and `hypothesis.class_id` on the detections topic
  (set `use_ppf_name_as_class_id:=false` to keep the old constant `object_id`)
- a **latched** `std_msgs/String` on `/perception/object_name`, so a node that
  starts late still learns which `.ply` to load
- `object_name.txt` in the results directory, and an `obj_name` key inside
  `detection_pem.json`, for consumers that read files rather than topics

A reply without `object_name` still parses, so an older server keeps working.

---

## Failure modes, and what they look like

| symptom | likely cause |
|---|---|
| every model rejected by the extent filter | depth is bad, or the mask grabbed background. The filter is then ignored and everything is scored, so the table is still readable. |
| correct part scores near zero **against itself** in the self-test | that model needs a finer per-model sampling step. `ppf_selftest.py` prints the exact `--add --sampling-step` command. `270circle`, a thin ring, is the example here. |
| two parts constantly swapping, tiny margin | they are genuinely similar from one view — e.g. `plate` (150×4×100) vs `test_objv1_base` (180×4×100). Raise `PPF_MIN_MARGIN`. |
| one model never wins anywhere | check its printed extents. Almost always an export-unit slip. |
| pose overlay is the right shape in the wrong place | classification was right, registration was not. Look at `vis_pose.png`. |
| pose overlay is the *wrong shape* | classification was wrong. Look at `classification.scores`. |

`vis_pose.png` in the results directory is the first artifact to open when anything
looks wrong: an oriented box of the wrong shape distinguishes a classification
failure from a registration failure immediately.

## Known limits

- Two thin plates of similar size are not separable from a single view. That is a
  property of the parts, not a defect; the margin is the signal.
- Thin curved parts need a finer sampling step than the 0.04 default.
- `test_objv2` and `test_objv3` have identical bounding boxes, so the extent filter
  cannot help — they separate on score alone (~0.8 vs ~0.05, cleanly).
- Classification adds ~1.4 s to a trigger on a 13-model library.
- The self-test is synthetic and uses exact CAD normals. Real depth does not.
  Treat its accuracy as an upper bound and validate against real masks.
