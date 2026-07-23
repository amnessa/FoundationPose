"""Point-pair-feature CAD *classifier*, built on OpenCV's `cv2.ppf_match_3d`.

The job is narrow: given the segmented point cloud SAM2 just handed us, say *which*
CAD model in the library it is. A name -- `test_objv2_ear` -- and nothing else.
FoundationPose then estimates the pose from that CAD, which is its job and it is
much better at it than PPF voting is.

That "and nothing else" is load-bearing, so to be unambiguous about what happens
to the transform PPF produces: a pose hypothesis is computed here, used *only* as
the argument to the verification score below, and then discarded. It is never
returned, never published, never handed to FoundationPose. It has to exist because
there is no way to ask "does this CAD explain this cloud?" without first placing
the CAD somewhere -- but it is scaffolding for the score, not an output.

Why a verification score at all, rather than just taking the model with the most
votes: **vote counts are not comparable across models.** Peak accumulator height
scales with model point count and with how self-similar the surface is. A large
`test_objv2_base` will out-vote a small `test_objv2_ear` almost regardless of which
one is actually in the mask. Votes are therefore used only to rank hypotheses
*within* one model, where the scale is shared and the comparison is meaningful.
Identity is decided by a two-sided normalized score:

    coverage  = |scene points within tau of the posed model| / |scene points|
    explained = |visible model points within tau of a scene point| / |visible model points|
    score     = coverage * explained

Coverage alone lets a big model win by blanketing the segment; explained alone lets
a small model win by hiding inside it. The product punishes both. This is
structurally Birdal & Ilic's Eq. 11 (3DV 2015) with the occlusion discount replaced
by an explicit visibility test, and it is the only number in this module that may
be compared between two different CAD models.

Relationship to the Birdal paper generally: the heavy machinery there -- the
coarse-to-fine depth segmentation of §2.2.2, the pose clustering and quaternion
averaging of §2.2.3 -- exists to turn PPF into a detector. SAM2 already gives us
exactly one object, and we discard the pose, so both are redundant here. What we
keep is the model representation (§2.1), the voting (§2.2.1, inside OpenCV) and the
hypothesis verification (§2.2.4), which is the part that makes the ranking sound.

A note on OpenCV, because it shapes the whole file: `PPF3DDetector` exposes
`trainModel` and `match`, and nothing else. In particular **the trained hashtable
cannot be serialized** -- there is no read/write in the bindings. So the .npz this
module writes is not the hashtable; it is everything expensive and fiddly that
happens *before* training (mesh loading, unit scaling, surface sampling, normals,
extents), and detectors are rebuilt from it at startup in a second or two per
model. Section "Persistence" below says so again where it matters.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

LIBRARY_FORMAT = 1


# ── Parameters ────────────────────────────────────────────────────────────

@dataclass
class PPFParams:
    """Build-time settings. Persisted into the .npz and restored on load, so that
    changing a default here can never silently invalidate a library on disk."""

    mesh_scale: float = 0.001       # PLY units -> metres. FreeCAD and BOP export mm,
                                    # and fp_server keeps everything in metres.
    n_model_points: int = 600       # target points per model after downsampling

    # OpenCV detector geometry. Both steps are *relative to each model's own
    # diameter* -- that is baked into cv2.ppf_match_3d and cannot be made absolute.
    # It is also the deeper reason vote counts do not compare across models: each
    # model is quantized on its own scale.
    relative_sampling_step: float = 0.04
    relative_distance_step: float = 0.04
    num_angles: int = 30

    def __post_init__(self):
        if not 0 < self.relative_sampling_step < 1:
            raise ValueError("relative_sampling_step must be in (0, 1)")


@dataclass
class QueryParams:
    """Per-request settings. Not persisted -- tune these against live scenes."""

    max_scene_points: int = 2000
    scene_sample_step: float = 1.0 / 5.0    # OpenCV: use every 5th scene point as S_r
    scene_relative_distance: float = 0.04

    n_hypotheses: int = 12          # top-voted poses verified per model. Votes rank
                                    # only within a model; see the module docstring.

    # Inlier radius for the verification score, in METRES, and deliberately not a
    # fraction of the model diameter. tau answers "is this scene point explained by
    # the posed CAD, within sensor noise" -- that is a property of the depth camera,
    # not of the model. Scaling it per-model would hand large models a wider
    # tolerance than small ones and quietly destroy the one property the score is
    # supposed to have, which is comparability across the library. 8mm suits a
    # RealSense at about 0.8m; tighten it for a better sensor or a closer standoff.
    tau_m: float = 0.008
    icp_polish: bool = True         # refine each hypothesis before scoring. Runs
                                    # identically on every candidate so it cannot bias
                                    # the ranking; it just stops PPF's coarse
                                    # quantization from showing up as a coverage
                                    # penalty that differs by model.
    icp_iterations: int = 50

    # Extent pre-filter. Nearly free, and on a library of differently-sized parts it
    # usually collapses the candidate set to one or two -- which is what stops added
    # latency from scaling as n_models * (match + verify).
    extent_filter: bool = True
    # The two bounds are deliberately asymmetric, because they are not the same kind
    # of claim. "Scene bigger than CAD" is physically impossible for a partial view
    # -- only depth noise, mask bleed onto the background, and the percentile
    # estimator itself can inflate it, and none of those reach 15%. "Scene smaller
    # than CAD" is routine: occlusion and grazing views shrink the visible extent
    # legitimately, so that side has to stay loose.
    #
    # This is worth getting right rather than setting generously. On the first real
    # frame tested, extent_over=0.30 let a 206mm CAD compete against a 255mm segment
    # -- 24% too small to be possible -- and it won on score, giving the wrong
    # answer. At 0.15 it is rejected outright and the correct model wins by 0.438.
    extent_over: float = 0.15
    extent_under: float = 0.55

    min_margin: float = 0.0         # below this top-to-runner-up gap the result is
                                    # flagged ambiguous (still returned; the caller
                                    # decides). Two thin plates of similar size are
                                    # genuinely indistinguishable from one view, and
                                    # the honest signal for that is a small margin.


# ── Model preparation ─────────────────────────────────────────────────────

# A part a robot picks up is somewhere between a centimetre and a metre across.
# Outside that range, after scaling, the export units were almost certainly not what
# `mesh_scale` assumed.
PLAUSIBLE_DIAMETER_M = (0.01, 1.5)


def load_mesh(path: str, mesh_scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh')
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)
    return mesh


def check_scale(name: str, mesh: trimesh.Trimesh, mesh_scale: float) -> Optional[str]:
    """Warn when a mesh's size makes no physical sense after scaling.

    `mesh_scale` is applied uniformly to every model in the library, but FreeCAD's
    export unit is a per-document setting -- one part exported in metres, or in
    centimetres, drops into an otherwise-millimetre library at 1000x or 10x the
    right size. That used to be merely wrong; now that the extent pre-filter rejects
    candidates on size, a mis-scaled model is silently unclassifiable: it is either
    thrown out of every scene it appears in, or it swallows scenes it has no
    business matching.

    The symptom is hard to read backwards from a bad classification, so we check it
    at index time, where the fix is obvious.
    """
    diameter = float(np.linalg.norm(mesh.extents))
    lo, hi = PLAUSIBLE_DIAMETER_M
    if lo <= diameter <= hi:
        return None
    # What scale *would* have been sensible, rounded to a familiar unit factor.
    raw = diameter / mesh_scale if mesh_scale else diameter
    suggestion = min([1.0, 0.01, 0.001], key=lambda s: abs(np.log(max(raw * s, 1e-9) / 0.15)))
    unit = {1.0: 'metres', 0.01: 'centimetres', 0.001: 'millimetres'}[suggestion]
    msg = (f"'{name}' is {diameter*1000:.1f}mm across after MESH_SCALE={mesh_scale} "
           f"-- outside the plausible {lo*1000:.0f}-{hi*1000:.0f}mm range. The file is "
           f"probably in {unit} (MESH_SCALE={suggestion}), not what was assumed. "
           f"The extent pre-filter compares physical size, so this model will not "
           f"classify correctly until the scale is right.")
    logging.warning(msg)
    return msg


def sample_model_cloud(mesh: trimesh.Trimesh, n_target: int,
                       seed: int = 0) -> np.ndarray:
    """Mesh -> the Nx6 (x,y,z,nx,ny,nz) float32 array OpenCV wants.

    We cannot use `cv2.ppf_match_3d.loadPLYSimple` here. These CAD files are
    FreeCAD boxes with eight vertices and no normals; loading them verbatim gives
    OpenCV eight unoriented points and there is nothing to match. The surface has to
    be sampled first.

    Sampling is uniform-by-area, then thinned by keeping one representative per
    voxel, with the voxel size bisected until the count lands near n_target. Keeping
    a *representative* rather than a per-voxel average is deliberate: averaging
    positions across a box edge rounds the corner off, and averaging normals across
    it produces a 45-degree normal belonging to no face at all. Every point returned
    sits exactly on the surface and carries its own face's normal -- which is why
    none of the MLS normal smoothing in the paper's §2.1.2 is needed. That exists to
    clean up scans; a CAD mesh already knows its normals exactly.
    """
    rng = np.random.default_rng(seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, max(60_000, n_target * 60),
                                                  seed=int(seed))
    pts = np.asarray(pts, dtype=np.float64)
    nrm = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    span = float(np.linalg.norm(mesh.extents))
    lo, hi, keep = span * 1e-3, span, None
    for _ in range(40):
        vox = np.sqrt(lo * hi)
        idx = np.floor((pts - pts.min(0)) / vox).astype(np.int64)
        _, first = np.unique(idx, axis=0, return_index=True)
        keep = first
        if len(first) > n_target * 1.1:
            lo = vox
        elif len(first) < n_target * 0.9:
            hi = vox
        else:
            break
        if hi / lo < 1.001:
            break
    keep = np.sort(keep)
    if len(keep) > n_target:
        keep = np.sort(rng.choice(keep, n_target, replace=False))
    return np.hstack([pts[keep], nrm[keep]]).astype(np.float32)


def cloud_diameter(pts: np.ndarray, n_sub: int = 1200, percentile: float = 99.5,
                   rng=None) -> float:
    """Robust diameter. The 99.5th percentile rather than the max, because one depth
    flyer would otherwise set the extent and swing the pre-filter."""
    if len(pts) < 2:
        return 0.0
    rng = rng or np.random.default_rng(0)
    if len(pts) > n_sub:
        pts = pts[rng.choice(len(pts), n_sub, replace=False)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(np.percentile(d[np.triu_indices(len(pts), 1)], percentile))


# ── The library ───────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    name: str                       # "test_objv2_ear" -- the stem, and what we report
    filename: str                   # "test_objv2_ear.ply"
    cloud: np.ndarray               # (P,6) float32, metres, in the CAD's own frame
    diameter: float                 # metres, max pairwise vertex distance
    extents: np.ndarray             # (3,) AABB extents, metres
    n_faces: int = 0
    # Per-model detector override. Thin curved parts (a 270-degree ring, say) need a
    # finer sampling step than a machined block; ppf_selftest.py is how you find out
    # which ones, and this is where you record the answer.
    sampling_step: Optional[float] = None
    # Set when the mesh's size looked implausible for the configured MESH_SCALE.
    # Not persisted -- it is re-derived on every index, and its whole job is to be
    # loud at the moment the model enters the library.
    scale_warning: Optional[str] = None
    detector: object = field(default=None, repr=False)

    def step(self, params: PPFParams) -> float:
        return self.sampling_step or params.relative_sampling_step


class PPFLibrary:
    def __init__(self, params: Optional[PPFParams] = None):
        self.params = params or PPFParams()
        self.models: List[ModelRecord] = []

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.models]

    def get(self, name: str) -> Optional[ModelRecord]:
        return next((m for m in self.models if m.name == name), None)

    def scale_outliers(self, factor: float = 5.0) -> List[Tuple[str, str]]:
        """Models whose size is wildly out of line with the rest of the library.

        `check_scale` catches the 1000x case -- a part exported in metres into a
        millimetre library -- because the result is implausible on its own. It cannot
        catch a 10x case: a 25mm part is a perfectly reasonable object in isolation,
        so nothing about it alone looks wrong.

        What does look wrong is a 25mm part sitting in a library where everything
        else is 100-360mm. That is the realistic failure -- FreeCAD's export unit is
        a per-document setting, so a single file drifts, not the whole set. Comparing
        against the library median catches it; the cost is that a library which is
        *genuinely* that diverse will produce a false positive, which is why this
        warns rather than refuses.

        The check nobody can automate is the one worth doing anyway:
        build_ppf_library.py prints every model's extents in millimetres, and you
        know what your parts actually measure.
        """
        sized = [m for m in self.models if m.diameter > 0]
        if len(sized) < 3:
            return []
        median = float(np.median([m.diameter for m in sized]))
        out = []
        for m in sized:
            ratio = m.diameter / median
            if ratio > factor or ratio < 1.0 / factor:
                out.append((m.name, (
                    f"'{m.name}' is {m.diameter*1000:.1f}mm across, {ratio:.2g}x the "
                    f"library median of {median*1000:.0f}mm. If that is not its real "
                    f"size, it was exported in different units from the rest.")))
        return out

    # --- construction ----------------------------------------------------

    def index_ply(self, path: str, sampling_step: Optional[float] = None) -> ModelRecord:
        p = self.params
        name = os.path.splitext(os.path.basename(path))[0]
        mesh = load_mesh(path, p.mesh_scale)
        scale_warning = check_scale(name, mesh, p.mesh_scale)
        cloud = sample_model_cloud(mesh, p.n_model_points)
        rec = ModelRecord(
            name=name, filename=os.path.basename(path), cloud=cloud,
            diameter=cloud_diameter(np.asarray(mesh.vertices)),
            extents=np.asarray(mesh.extents, dtype=np.float64),
            n_faces=len(mesh.faces), sampling_step=sampling_step,
            scale_warning=scale_warning)
        logging.info(f"  {name}: {len(cloud)} pts, diam={rec.diameter*1000:.0f}mm, "
                     f"extents={np.round(rec.extents*1000, 1).tolist()}mm")
        return rec

    @classmethod
    def build(cls, ply_paths: Sequence[str],
              params: Optional[PPFParams] = None) -> "PPFLibrary":
        lib = cls(params)
        logging.info(f"indexing {len(ply_paths)} CAD file(s)")
        for path in ply_paths:
            try:
                lib.models.append(lib.index_ply(path))
            except Exception as exc:      # one unreadable PLY must not sink the library
                logging.error(f"  {os.path.basename(path)}: FAILED to index ({exc})")
        return lib

    def add_model(self, path: str, sampling_step: Optional[float] = None,
                  train: bool = True, replace: bool = True) -> ModelRecord:
        """Index one new CAD file into the library.

        This is the hook for CAD that shows up at runtime -- a part composed from two
        existing models and exported as a new .ply, for instance. Nothing about the
        rest of the library is touched: each model owns its own detector, so adding
        one is genuinely incremental. The caller is responsible for `save()`ing
        afterwards if the addition should outlive the process.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        existing = next((i for i, m in enumerate(self.models) if m.name == name), None)
        if existing is not None and not replace:
            raise ValueError(f"model '{name}' is already in the library")
        rec = self.index_ply(path, sampling_step=sampling_step)
        if train:
            self._train_one(rec)
        if existing is not None:
            self.models[existing] = rec
        else:
            self.models.append(rec)
        return rec

    # --- training --------------------------------------------------------

    def _train_one(self, rec: ModelRecord) -> None:
        t0 = time.monotonic()
        det = cv2.ppf_match_3d_PPF3DDetector(
            rec.step(self.params), self.params.relative_distance_step,
            self.params.num_angles)
        det.trainModel(rec.cloud)
        rec.detector = det
        logging.info(f"  trained {rec.name} ({time.monotonic()-t0:.1f}s)")

    def train(self, names: Optional[Sequence[str]] = None) -> None:
        """Build the OpenCV hashtables. ~1-2s per model, hence `lazy` at the call site.

        This exists as a separate step from `load()` only because it cannot be
        avoided: OpenCV has no way to serialize a trained PPF3DDetector, so the
        hashtable has to be rebuilt in-process every time the server starts.
        """
        todo = [m for m in self.models
                if (names is None or m.name in names) and m.detector is None]
        if not todo:
            return
        logging.info(f"training {len(todo)} PPF detector(s)")
        for rec in todo:
            self._train_one(rec)

    # --- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Write the library to a .npz.

        To be explicit about what is and is not in this file: it holds the *sampled
        oriented point clouds* and their metadata, not the PPF hashtable. OpenCV
        cannot serialize a trained detector. What the file saves you is the slow and
        version-sensitive half of the offline stage -- mesh loading, unit scaling,
        surface sampling, normal assignment, extent measurement -- and it guarantees
        every process that loads it sees byte-identical model clouds. Training still
        runs at startup.
        """
        if not self.models:
            raise ValueError("refusing to save an empty library")
        counts = [len(m.cloud) for m in self.models]
        np.savez_compressed(
            path,
            format=np.int64(LIBRARY_FORMAT),
            params=json.dumps(asdict(self.params)),
            names=np.array([m.name for m in self.models]),
            filenames=np.array([m.filename for m in self.models]),
            clouds=np.concatenate([m.cloud for m in self.models]),
            offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            diameters=np.array([m.diameter for m in self.models]),
            extents=np.array([m.extents for m in self.models]).reshape(-1, 3),
            n_faces=np.array([m.n_faces for m in self.models]),
            sampling_steps=np.array([m.sampling_step if m.sampling_step else np.nan
                                     for m in self.models]),
        )
        logging.info(f"saved PPF library -> {path} "
                     f"({len(self.models)} models, {os.path.getsize(path)/1e6:.1f} MB)")

    @classmethod
    def load(cls, path: str, train: bool = True) -> "PPFLibrary":
        z = np.load(path, allow_pickle=False)
        fmt = int(z['format']) if 'format' in z else 0
        if fmt != LIBRARY_FORMAT:
            raise ValueError(f"{path} is a format-{fmt} PPF library; this code writes "
                             f"format {LIBRARY_FORMAT}. Rebuild it with build_ppf_library.py.")
        # Stored params win over the dataclass defaults: the detector must be built
        # exactly the way the clouds were sampled, whatever the defaults are today.
        lib = cls(PPFParams(**json.loads(str(z['params']))))
        off = z['offsets']
        for i, name in enumerate(z['names']):
            step = float(z['sampling_steps'][i])
            lib.models.append(ModelRecord(
                name=str(name), filename=str(z['filenames'][i]),
                cloud=z['clouds'][int(off[i]):int(off[i + 1])],
                diameter=float(z['diameters'][i]), extents=z['extents'][i],
                n_faces=int(z['n_faces'][i]),
                sampling_step=None if np.isnan(step) else step))
        logging.info(f"loaded PPF library {path}: {lib.names}")
        if train:
            lib.train()
        return lib

    # --- classification --------------------------------------------------

    def classify(self, scene_pts: np.ndarray, scene_nrm: np.ndarray,
                 K: Optional[np.ndarray] = None,
                 q: Optional[QueryParams] = None, seed: int = 0) -> Dict:
        """Rank every model against one segmented cloud.

        Returns a report dict. The whole score table is always in it, not just the
        winner: the runner-up margin is the number worth watching during bring-up,
        and carrying it costs nothing.
        """
        q = q or QueryParams()
        rng = np.random.default_rng(seed)
        t0 = time.monotonic()

        if not self.models:
            return {'ok': False, 'message': 'the PPF library is empty', 'scores': {}}
        if len(scene_pts) < 20:
            return {'ok': False, 'scores': {},
                    'message': f'only {len(scene_pts)} scene points survived masking'}

        if len(scene_pts) > q.max_scene_points:
            sel = np.sort(rng.choice(len(scene_pts), q.max_scene_points, replace=False))
            scene_pts, scene_nrm = scene_pts[sel], scene_nrm[sel]
        scene = np.hstack([scene_pts, scene_nrm]).astype(np.float32)
        scene_tree = cKDTree(scene_pts)
        scene_diam = cloud_diameter(scene_pts, rng=rng)

        # --- extent pre-filter -------------------------------------------
        rejected: Dict[str, str] = {}
        candidates = list(self.models)
        if q.extent_filter:
            kept = []
            for m in candidates:
                if m.diameter <= 0:
                    kept.append(m)
                    continue
                ratio = scene_diam / m.diameter
                if ratio > 1.0 + q.extent_over:
                    rejected[m.name] = (f"scene {scene_diam*1000:.0f}mm exceeds CAD "
                                        f"{m.diameter*1000:.0f}mm by {(ratio-1)*100:.0f}%")
                elif ratio < 1.0 - q.extent_under:
                    rejected[m.name] = (f"scene {scene_diam*1000:.0f}mm is only "
                                        f"{ratio*100:.0f}% of CAD {m.diameter*1000:.0f}mm")
                else:
                    kept.append(m)
            # Never return nothing on the strength of a heuristic. If the filter
            # rejected everything the depth is probably bad, and a full scored
            # ranking is far more diagnostic than an empty result.
            if kept:
                candidates = kept
            else:
                logging.warning("extent pre-filter rejected every model; ignoring it")
                rejected = {}

        self.train([m.name for m in candidates])

        # --- match + verify ----------------------------------------------
        icp = cv2.ppf_match_3d_ICP(q.icp_iterations, 0.01, 2.5, 8) if q.icp_polish else None
        results = []
        for m in candidates:
            results.append(self._score_model(m, scene, scene_pts, scene_tree, K, icp, q))
        for m in self.models:
            if m.name in rejected:
                results.append({'name': m.name, 'score': 0.0, 'coverage': 0.0,
                                'explained': 0.0, 'votes': 0.0, 'visible_points': 0,
                                'diameter_mm': round(m.diameter * 1000, 1),
                                'rejected': rejected[m.name]})
        results.sort(key=lambda r: r['score'], reverse=True)

        top = results[0]
        runner_up = results[1]['score'] if len(results) > 1 else 0.0
        margin = top['score'] - runner_up
        model = self.get(top['name'])
        return {
            'ok': top['score'] > 0.0,
            'object_name': top['name'] if top['score'] > 0 else None,
            'object_file': model.filename if (model and top['score'] > 0) else None,
            'score': round(top['score'], 4),
            'margin': round(margin, 4),
            'ambiguous': bool(margin < q.min_margin),
            'runner_up': results[1]['name'] if len(results) > 1 else None,
            'scores': {r['name']: round(r['score'], 4) for r in results},
            'detail': {r['name']: r for r in results},
            'rejected': rejected,
            'scene_points': int(len(scene_pts)),
            'scene_diameter_mm': round(scene_diam * 1000, 1),
            'elapsed_sec': round(time.monotonic() - t0, 2),
        }

    def _score_model(self, m: ModelRecord, scene: np.ndarray, scene_pts: np.ndarray,
                     scene_tree: cKDTree, K, icp, q: QueryParams) -> Dict:
        out = {'name': m.name, 'score': 0.0, 'coverage': 0.0, 'explained': 0.0,
               'votes': 0.0, 'visible_points': 0,
               'diameter_mm': round(m.diameter * 1000, 1)}
        tau = q.tau_m
        try:
            poses = m.detector.match(scene, q.scene_sample_step, q.scene_relative_distance)
        except cv2.error as exc:
            out['error'] = str(exc).splitlines()[-1][:200]
            return out
        if not poses:
            return out

        for p in poses[:q.n_hypotheses]:
            T = np.asarray(p.pose, dtype=np.float64)
            if icp is not None:
                T = _icp_refine(icp, m.cloud, T, scene)
            score, cov, exp, n_vis = _verification_score(m.cloud, T, scene_pts,
                                                         scene_tree, K, tau)
            if score > out['score']:
                out.update(score=float(score), coverage=float(cov),
                           explained=float(exp), visible_points=int(n_vis),
                           votes=float(p.numVotes))
        return out


# ── Verification (Birdal & Ilic §2.2.4, two-sided form) ───────────────────

def _icp_refine(icp, cloud: np.ndarray, T: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Polish one hypothesis before scoring it.

    OpenCV's `registerModelToScene` is plain ICP -- it assumes the model is already
    roughly on the scene -- so the model must be transformed by the PPF pose first
    and the returned correction composed on the left. Refining the *unposed* cloud
    (an easy mistake) silently gives you a meaningless residual.

    Every candidate gets exactly the same treatment, so this cannot bias the
    ranking. What it removes is PPF's coarse angular quantization, which would
    otherwise land in the score as a coverage penalty that differs by model.
    """
    posed = np.hstack([cloud[:, :3] @ T[:3, :3].T + T[:3, 3],
                       cloud[:, 3:] @ T[:3, :3].T]).astype(np.float32)
    try:
        _, _, correction = icp.registerModelToScene(posed, scene)
        return np.asarray(correction, dtype=np.float64) @ T
    except cv2.error:
        return T


def _visible_model_points(pts_cam: np.ndarray, nrm_cam: np.ndarray,
                          K: Optional[np.ndarray], tau: float) -> np.ndarray:
    """Which points of the posed model a camera at the origin could actually see.

    Two tests, cheapest first:

      backface culling   n . (c - p) > 0 with the camera at the origin. Exact for the
                         convex plates and blocks that dominate this library.
      coarse z-buffer    bin the projection into cells about tau wide at the object's
                         depth, keep only points within tau of the nearest surface in
                         their cell. This is what catches self-occlusion on a
                         non-convex part. The cells are deliberately much larger than
                         a pixel: a per-pixel buffer over a 600-point cloud would
                         almost never see two points collide, and so would test
                         nothing at all.
    """
    front = np.einsum('ij,ij->i', nrm_cam, -pts_cam) > 0
    z = pts_cam[:, 2]
    front &= z > 1e-6
    if K is None or front.sum() < 8:
        return front

    cell_px = max(1.0, float(K[0, 0]) * tau / max(float(np.median(z[front])), 1e-6))
    u = np.floor((K[0, 0] * pts_cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]) / cell_px)
    v = np.floor((K[1, 1] * pts_cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]) / cell_px)
    idx = np.where(front)[0]
    cu = (u - u[idx].min()).astype(np.int64)
    cv_ = (v - v[idx].min()).astype(np.int64)
    cell = cv_ * (int(cu[idx].max()) + 1) + cu

    nearest = np.full(int(cell[idx].max()) + 1, np.inf)
    np.minimum.at(nearest, cell[idx], z[idx])
    visible = np.zeros(len(pts_cam), dtype=bool)
    visible[idx] = z[idx] <= nearest[cell[idx]] + tau
    return visible


def _verification_score(cloud: np.ndarray, T: np.ndarray, scene_pts: np.ndarray,
                        scene_tree: cKDTree, K: Optional[np.ndarray], tau: float
                        ) -> Tuple[float, float, float, int]:
    """score = coverage * explained. The one number comparable between CAD models."""
    P = cloud[:, :3] @ T[:3, :3].T + T[:3, 3]
    N = cloud[:, 3:] @ T[:3, :3].T
    vis = _visible_model_points(P, N, K, tau)
    if vis.sum() < 4:
        return 0.0, 0.0, 0.0, int(vis.sum())
    Pv = P[vis]

    explained = float((scene_tree.query(Pv, workers=-1)[0] < tau).mean())
    coverage = float((cKDTree(Pv).query(scene_pts, workers=-1)[0] < tau).mean())
    return coverage * explained, coverage, explained, int(vis.sum())


# ── Scene preparation ─────────────────────────────────────────────────────

def scene_cloud_from_mask(depth: np.ndarray, K: np.ndarray, mask: np.ndarray,
                          max_points: int = 2000, voxel: Optional[float] = None,
                          erode_px: int = 2, normal_dilate_px: int = 4,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """SAM2 mask + depth -> the small clean oriented cloud the classifier queries with.

    Three details here matter more than the projection itself:

    * **The mask is eroded first.** Depth pixels straddling the silhouette mix the
      object with whatever is behind it, and those mixed points form a skirt of
      invented geometry that drags every verification score down -- unevenly, since
      it hurts thin parts most.

    * **Normals are estimated on a dilated crop, then everything outside the eroded
      mask is dropped.** A point on the boundary has neighbours on one side only, so
      its normal tips inward. Fitting over a slightly larger crop and then discarding
      the border gives every surviving point a complete neighbourhood. PPF is built
      entirely out of normals, so this is not cosmetic.

    * **Normals are flipped toward the camera**, matching the outward-facing face
      normals of the CAD models. The angle features are sign-sensitive; a flipped
      scene normal lands the pair in a different bucket entirely.
    """
    mask = mask.astype(bool)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1,) * 2, np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), k) > 0
        if eroded.sum() >= 50:          # never erode a small mask out of existence
            mask = eroded
    wide = mask
    if normal_dilate_px > 0:
        k = np.ones((2 * normal_dilate_px + 1,) * 2, np.uint8)
        wide = cv2.dilate(mask.astype(np.uint8), k) > 0

    valid = wide & (depth > 1e-3) & np.isfinite(depth)
    vs, us = np.nonzero(valid)
    if len(us) < 20:
        return np.zeros((0, 3)), np.zeros((0, 3))
    z = depth[vs, us].astype(np.float64)
    pts = np.stack([(us - K[0, 2]) * z / K[0, 0],
                    (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
    inner = mask[vs, us]
    if inner.sum() < 20:
        return np.zeros((0, 3)), np.zeros((0, 3))

    if voxel is None:
        voxel = max(cloud_diameter(pts[inner]) / 60.0, 0.0015)

    # OpenCV's own normal estimator, with the viewpoint set to the camera origin so
    # it orients them for us.
    _, with_normals = cv2.ppf_match_3d.computeNormalsPC3d(
        pts.astype(np.float32), 12, True, (0.0, 0.0, 0.0))
    nrm = np.asarray(with_normals)[:, 3:].astype(np.float64)

    pts, nrm = pts[inner], nrm[inner]
    idx = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    _, first = np.unique(idx, axis=0, return_index=True)
    pts, nrm = pts[first], nrm[first]
    if len(pts) > max_points:
        sel = np.sort(np.random.default_rng(seed).choice(len(pts), max_points, replace=False))
        pts, nrm = pts[sel], nrm[sel]
    return pts, nrm


def find_ply_files(directory: str) -> List[str]:
    """Every .ply in a directory, sorted. The offline stage's input."""
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith('.ply'))
