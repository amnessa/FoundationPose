#!/usr/bin/env python3
"""Offline stage: scan a directory of CAD .ply files and write the PPF library .npz.

    python scripts/build_ppf_library.py
    python scripts/build_ppf_library.py --cad-dir Data/Input --out Data/ppf_library.npz
    python scripts/build_ppf_library.py --add Data/Input/new_part.ply

Run this once per new part. fp_server loads the .npz at startup (and rebuilds it
automatically if it is missing), so the only reason to run it by hand is to add or
re-tune a model without restarting the server.

What the .npz actually contains, since the name "hash table" is the obvious guess
and is wrong: the *sampled oriented point clouds* and their metadata -- not the PPF
hashtable. OpenCV's PPF3DDetector has no serialization in its bindings, so the
hashtable is rebuilt in-process at startup (1-2s per model). What this file saves
you is the slow, fiddly, version-sensitive half of the offline stage -- mesh
loading, mm-to-metre scaling, surface sampling, normal assignment, extent
measurement -- and it guarantees every process that loads it sees byte-identical
model clouds, which is what makes scores reproducible across restarts.
"""

import argparse
import logging
import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'scripts'))

from ppf_classifier import PPFLibrary, PPFParams, find_ply_files  # noqa: E402

DEFAULT_CAD_DIR = os.path.join(CODE_DIR, "Data", "Input")
DEFAULT_OUT = os.path.join(CODE_DIR, "Data", "ppf_library.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cad-dir', default=DEFAULT_CAD_DIR,
                    help='directory scanned for .ply files (default: %(default)s)')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='library .npz to write (default: %(default)s)')
    ap.add_argument('--add', metavar='PLY', action='append', default=[],
                    help='add/replace one .ply in an existing library instead of '
                         'rebuilding from the directory; repeatable')
    ap.add_argument('--sampling-step', type=float,
                    help='per-model relative sampling step for --add. Thin curved '
                         'parts need a finer step (try 0.025) than machined blocks.')
    ap.add_argument('--mesh-scale', type=float, default=PPFParams.mesh_scale,
                    help='PLY units -> metres (default: %(default)s, i.e. mm)')
    ap.add_argument('--model-points', type=int, default=PPFParams.n_model_points,
                    help='target sampled points per model (default: %(default)s)')
    ap.add_argument('--relative-sampling-step', type=float,
                    default=PPFParams.relative_sampling_step,
                    help='library-wide PPF sampling step (default: %(default)s)')
    ap.add_argument('--num-angles', type=int, default=PPFParams.num_angles,
                    help='angle quantization bins (default: %(default)s)')
    ap.add_argument('--no-train', action='store_true',
                    help='skip the training smoke test (faster, less checked)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.add:
        if not os.path.exists(args.out):
            logging.error(f"--add needs an existing library; {args.out} is not there. "
                          "Build it first without --add.")
            return 1
        lib = PPFLibrary.load(args.out, train=False)
        for path in args.add:
            if not os.path.exists(path):
                logging.error(f"no such file: {path}")
                return 1
            lib.add_model(path, sampling_step=args.sampling_step, train=not args.no_train)
    else:
        paths = find_ply_files(args.cad_dir)
        if not paths:
            logging.error(f"no .ply files in {args.cad_dir}")
            return 1
        params = PPFParams(
            mesh_scale=args.mesh_scale,
            n_model_points=args.model_points,
            relative_sampling_step=args.relative_sampling_step,
            relative_distance_step=args.relative_sampling_step,
            num_angles=args.num_angles)
        lib = PPFLibrary.build(paths, params)
        if not lib.models:
            logging.error("nothing could be indexed")
            return 1
        if not args.no_train:
            # Training here is a smoke test, not a saving: the result cannot be
            # serialized, so the server trains again on startup. It is still worth
            # doing -- a mesh that makes OpenCV throw should fail here, offline,
            # rather than on the first live trigger.
            lib.train()

    lib.save(args.out)

    print(f"\n{len(lib.models)} model(s) in {args.out}:")
    for m in sorted(lib.models, key=lambda r: -r.diameter):
        step = f"  step={m.sampling_step}" if m.sampling_step else ""
        print(f"  {m.name:24s} {len(m.cloud):5d} pts  diam={m.diameter*1000:6.1f}mm  "
              f"extents={np.round(np.asarray(m.extents)*1000, 1).tolist()}mm{step}")

    # Same-size pairs cannot be separated by the extent pre-filter, so they are the
    # ones that will lean entirely on the verification score -- worth knowing about
    # before they surprise you on a live trigger.
    # A mis-scaled model is worse than a wrong one: the extent pre-filter compares
    # physical size, so it will be thrown out of every scene it belongs in. FreeCAD's
    # export unit is per-document, so this happens by accident.
    warnings = [m.scale_warning for m in lib.models if m.scale_warning]
    warnings += [msg for _, msg in lib.scale_outliers()]
    if warnings:
        print(f"\n*** SCALE WARNING (MESH_SCALE={lib.params.mesh_scale}) ***")
        for msg in warnings:
            print(f"  {msg}")
        print("  Re-export at a consistent unit. Note --mesh-scale applies to EVERY "
              "model, so a\n  single odd file has to be re-exported rather than "
              "worked around here.")
    else:
        print(f"\nScale check passed (MESH_SCALE={lib.params.mesh_scale}). Still worth "
              "a glance:\n  the extents above are millimetres, and you know what your "
              "parts actually measure.\n  A 10x export-unit slip on one file is not "
              "detectable any other way, and it makes\n  that model unclassifiable — "
              "the extent pre-filter compares physical size.")

    close = [(a.name, b.name, a.diameter)
             for i, a in enumerate(lib.models) for b in lib.models[i + 1:]
             if a.diameter > 0 and abs(a.diameter - b.diameter) / a.diameter < 0.10]
    if close:
        print("\nmodels within 10% of each other in diameter -- the extent pre-filter "
              "cannot separate these,\nso watch the score margin when one of them is "
              "the target:")
        for a, b, d in close:
            print(f"  {a} <-> {b}  (~{d*1000:.0f}mm)")
    print("\nNext: python scripts/ppf_selftest.py --library " + args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
