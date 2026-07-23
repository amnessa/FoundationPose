# PPF CAD classification

The server used to be told which CAD model was on the table, via `MESH_PATH`. It
now works that out per request: SAM2 gives a mask, the masked depth becomes a
small oriented point cloud, and a point-pair-feature classifier matches that cloud
against a library built from every `.ply` in `CAD_DIR`. The winner *names* the CAD
that FoundationPose registers against.

```
 RGB-D  ──▶ SAM2 ──▶ mask ──▶ masked depth ──▶ point cloud
                                                    │
                                        PPF classifier (OpenCV surface_matching)
                                                    │
                                          "test_objv1_base"          ◀── a name, only
                                                    │
                                    FoundationPose.register(that .ply)
                                                    │
                                        pose (4x4, metres) ──▶ bridge ──▶ ICP client
```

## PPF names the object; it does not pose it

Pose estimation is FoundationPose's job and it is much better at it than PPF
voting. The classifier does compute a transform internally — there is no way to
ask "does this CAD explain this cloud?" without first placing the CAD somewhere —
but that transform is used *only* to compute the score below and is then dropped.
It is never returned, never published, and never handed to FoundationPose, which
re-estimates the pose from scratch on the CAD it is given.

## Why votes cannot decide the winner

Peak accumulator height scales with model point count and with how self-similar
the surface is, so it is not comparable between two different CAD models. Votes
are therefore used only to rank hypotheses *within* one model, where the scale is
shared. Identity is decided by a two-sided normalized score:

```
coverage  = |scene points within tau of the posed model| / |scene points|
explained = |visible model points within tau of a scene point| / |visible model points|
score     = coverage * explained
```

Coverage alone lets a large model win by blanketing the segment; explained alone
lets a small model win by hiding inside it. The product punishes both. This is
structurally Birdal & Ilic's Eq. 11 (3DV 2015) with the occlusion discount
replaced by an explicit visibility test (backface culling plus a coarse z-buffer).

`tau` is **absolute metres** (`PPF_TAU`, default 8 mm), not a fraction of the model
diameter. It represents depth-sensor noise, which is a property of the camera, not
of the CAD. Making it model-relative hands large models a wider tolerance than
small ones and destroys the one property the score is supposed to have —
comparability across the library. This was measured: switching `tau` from
model-relative to absolute moved the self-test from 65 % to 85 %.

## The extent pre-filter, and why its upper bound is tight

Before any matching, models whose diameter is incompatible with the segment's are
rejected outright. It is nearly free, and on the real test frame it eliminated 8 of
13 candidates — which is what keeps latency from scaling with library size.

The bounds are asymmetric because they are not the same kind of claim:

- **`extent_over` (15 %)** — "the scene is bigger than the CAD" is *physically
  impossible* for a partial view. Only depth noise, mask bleed onto the background,
  and the percentile estimator can inflate it, and none of those reach 15 %.
- **`extent_under` (55 %)** — "the scene is smaller than the CAD" is routine.
  Occlusion and grazing views shrink the visible extent legitimately, so this side
  must stay loose.

This is worth getting right rather than setting generously, and it was originally
set wrong. At `extent_over = 0.30`, a 206 mm CAD (`test_objv1_base`) was allowed to
compete against a 255 mm segment — 24 % too small to be possible — and it won on
score, giving the wrong answer on the one real frame available. At 0.15 it is
rejected outright and the correct model (`test_objv2_ear`) wins by 0.438 instead of
losing by 0.049.

The general lesson: the verification score is good at telling similar shapes apart,
but it is not a substitute for a hard physical constraint. Enforce the constraint
first.

If the filter rejects everything, it is ignored and every model is scored — a full
ranking is far more diagnostic than an empty result.

## Offline: building the library

```bash
python scripts/build_ppf_library.py                      # scan CAD_DIR -> Data/ppf_library.npz
python scripts/build_ppf_library.py --add new_part.ply   # add one model
```

`fp_server` builds the library automatically if the `.npz` is missing, so this is
only needed to add or re-tune a model without a restart.

**What the `.npz` contains** — the name "hash table" is the obvious guess and it is
wrong. OpenCV's `PPF3DDetector` exposes only `trainModel` and `match`; the trained
hashtable **cannot be serialized**. The file holds the *sampled oriented point
clouds* and their metadata: the slow, version-sensitive half of the offline stage
(mesh loading, mm→m scaling, surface sampling, normal assignment, extent
measurement). Detectors are rebuilt in-process at startup, ~1–2 s per model.

### Units: FreeCAD's export scale is per-document

`MESH_SCALE` (default `0.001`, mm→m) is applied uniformly to **every** model in the
library. FreeCAD's export unit is a per-document setting, so a single part exported
in metres or centimetres lands in an otherwise-millimetre library at 1000× or 10×
the wrong size.

This used to be merely inaccurate. Now that the extent pre-filter rejects candidates
on physical size, a mis-scaled model is *silently unclassifiable* — thrown out of
every scene it belongs in, or swallowing scenes it has no business matching. The
symptom is very hard to read backwards from a bad classification, so it is checked
at index time:

- **absolute plausibility** — a part outside 10 mm–1.5 m after scaling is flagged,
  with the scale that would have been sensible. This catches the 1000 × case.
- **library-relative outliers** — a model more than 5× off the library median is
  flagged. This catches the realistic case: one drifted file among correct ones.
- **neither catches a 10× slip on an isolated part**, because a 25 mm object is
  perfectly reasonable on its own. `build_ppf_library.py` prints every model's
  extents in millimetres for exactly this reason — you know what your parts
  measure, and a glance at that table is the only reliable check.

`--mesh-scale` applies to the whole library, so one odd file has to be re-exported
rather than worked around at build time.

The CAD files here are FreeCAD boxes with eight vertices and no normals, so
`cv2.ppf_match_3d.loadPLYSimple` is not usable — the surface is sampled with
trimesh first, keeping one representative point per voxel so every point sits
exactly on the surface and carries its own face's exact normal.

## Checking the library before you trust it

```bash
python scripts/ppf_selftest.py --views 3
```

Renders synthetic partial views of every model and classifies each against the
whole library. Two things to read off it:

- **A low diagonal** (a model failing against *itself*) means that part needs a
  finer per-model sampling step — the script prints the exact `--add` command.
  `270circle`, a thin 270° ring, is the example in this library.
- **A small margin** between two parts is information about your CAD set, not a
  defect. `plate` (150×4×100) and `test_objv1_base` (180×4×100) are two thin
  rectangles differing by 30 mm on one edge; from a single partial view they are
  near-indistinguishable, and the classifier says so.

Treat the number as an **upper bound** — synthetic views carry exact CAD normals,
real depth does not, and PPF is built entirely out of normals. The one real frame
tested classifies correctly (`test_objv2_ear`, margin 0.438), but one frame is not
an evaluation. Collect real masks and run `/classify` against them before trusting
any threshold.

## Handling ambiguity

`PPF_MIN_MARGIN` flags a result whose top-to-runner-up gap is too small;
`PPF_STRICT=1` makes the server refuse (HTTP 422) rather than guess. Both default
to permissive, so watch real margins first. The full score table is in every reply
regardless — the runner-up gap costs nothing to carry and is the number worth
watching during bring-up.

## Adding a CAD model at runtime

For a part composed from existing models and exported as a new mesh:

```bash
curl -F model=@new_part.ply http://host:5000/add_model
```

or from the bridge, without a restart or a custom service type:

```bash
ros2 param set /foundationpose_bridge model_ply_path /path/to/new_part.ply
ros2 service call /foundationpose_bridge/add_model std_srvs/srv/Trigger
```

Each model owns its own detector, so adding one retrains nothing else. The reply
lists any existing models within 10 % of the new one's diameter — the extent
pre-filter cannot separate those, so they will now compete on score alone.

## Passing the name downstream

The `/predict_pose` reply gains `object_name`, `object_file` and the full
`classification` block. The bridge republishes the name three ways so a tracking
node can load the right `.ply`:

- `Detection3D.id` and `ObjectHypothesisWithPose.hypothesis.class_id`
  (set `use_ppf_name_as_class_id:=false` to keep the old constant `object_id`)
- a latched `std_msgs/String` on `/perception/object_name`
- `object_name.txt` in the results directory, plus an `obj_name` key inside
  `detection_pem.json`, for consumers that read files rather than topics

A reply without `object_name` still parses, so an older server keeps working.

## Endpoints

| endpoint | purpose |
|---|---|
| `POST /classify` | classification only, no GPU pose. **Use this during bring-up** — ~1.4 s and returns the whole score table. |
| `POST /predict_pose` | the full pipeline: mask → classify → register → pose |
| `POST /add_model` | index a new `.ply` into the live library and persist it |
| `GET /health` | what is loaded, and what is in the library |

## Configuration

| env var | default | meaning |
|---|---|---|
| `PPF_ENABLE` | `1` | `0` pins the server to `MESH_PATH`, as before |
| `CAD_DIR` | `Data/Input` | directory scanned for library `.ply` files |
| `PPF_LIBRARY` | `Data/ppf_library.npz` | the library file |
| `PPF_TAU` | `0.008` | verification inlier radius in metres — your depth noise |
| `MESH_SCALE` | `0.001` | CAD units → metres, applied to every model (see Units above) |
| `PPF_MIN_MARGIN` | `0.0` | margin below which a result is flagged ambiguous |
| `PPF_STRICT` | `0` | `1` refuses ambiguous/failed classification instead of falling back |
| `MESH_PATH` | `test_objv2_ear.ply` | fallback CAD when PPF is off or finds nothing |

## Known limits

- Two thin plates of similar size are not separable from one view. This is a
  property of the parts; the margin is the signal.
- Thin curved parts (`270circle`) need a finer per-model sampling step than the
  0.04 default. `ppf_selftest.py` finds them.
- `test_objv2` and `test_objv3` have identical bounding boxes, so the extent
  pre-filter cannot help; they separate on score alone (~0.8 vs ~0.05, cleanly).
- Classification adds ~1.4 s to a trigger on a 13-model library.
