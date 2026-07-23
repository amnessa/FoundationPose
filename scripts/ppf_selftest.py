#!/usr/bin/env python3
"""Synthetic confusion matrix for the PPF library.

    python scripts/ppf_selftest.py
    python scripts/ppf_selftest.py --views 4 --noise 0.003

For every model in the library this renders a synthetic partial view of it -- a
random rotation, backface culling, Gaussian depth noise -- and runs the full
classifier against the whole library. The diagonal should win.

Run it after every rebuild, and especially after `--add`ing a new part. It is the
cheapest way to find out two things before a live trigger does:

  * which models the library cannot separate from each other (look at the margin
    column; two thin plates of similar size will be near-tied, and that is
    information about your parts, not a bug), and
  * which models fail against *themselves* -- a low diagonal score means that part
    needs a finer per-model sampling step. Rebuild it with
    `build_ppf_library.py --add <ply> --sampling-step 0.025`.

The one thing this cannot tell you is how the classifier behaves on real depth.
Synthetic views carry exact CAD normals; a real RealSense cloud does not, and PPF
is built entirely out of normals. Treat the numbers here as an upper bound.
"""

import argparse
import logging
import os
import sys

import numpy as np
import trimesh

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'scripts'))

from ppf_classifier import PPFLibrary, QueryParams  # noqa: E402

DEFAULT_LIBRARY = os.path.join(CODE_DIR, "Data", "ppf_library.npz")
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def synthetic_view(cloud: np.ndarray, seed: int, noise: float, distance: float):
    """One partial view of a model cloud: random pose, backface cull, add noise."""
    rng = np.random.default_rng(seed)
    R = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
    t = np.array([0.02, -0.01, distance])
    P = cloud[:, :3] @ R.T + t
    N = cloud[:, 3:] @ R.T
    vis = np.einsum('ij,ij->i', N, -P) > 0
    P, N = P[vis], N[vis]
    return P + rng.normal(0, noise, P.shape), N


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--library', default=DEFAULT_LIBRARY)
    ap.add_argument('--views', type=int, default=3, help='random views per model')
    ap.add_argument('--noise', type=float, default=0.0015, help='depth noise sigma [m]')
    ap.add_argument('--distance', type=float, default=0.8, help='camera distance [m]')
    ap.add_argument('--only', action='append', default=[], help='restrict to these models')
    ap.add_argument('--no-extent-filter', action='store_true')
    ap.add_argument('--quiet', action='store_true', help='hide per-model INFO logs')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format='%(levelname)s %(message)s')

    if not os.path.exists(args.library):
        print(f"no library at {args.library}; run build_ppf_library.py first")
        return 1
    lib = PPFLibrary.load(args.library, train=True)
    q = QueryParams(extent_filter=not args.no_extent_filter)

    targets = [m for m in lib.models if not args.only or m.name in args.only]
    names = lib.names
    width = max(11, min(13, max(len(n) for n in names) + 1))

    print(f"\n{len(targets)} model(s) x {args.views} view(s), noise={args.noise*1000:.1f}mm\n")
    print(f"{'truth':<22}" + "".join(f"{n[:width-1]:>{width}}" for n in names)
          + f"{'winner':>24}{'margin':>8}")

    n_ok = n_total = 0
    weak_diagonal, confusions = [], []
    for m in targets:
        for v in range(args.views):
            pts, nrm = synthetic_view(m.cloud, seed=1000 + v, noise=args.noise,
                                      distance=args.distance)
            rep = lib.classify(pts, nrm, K=K, q=q)
            scores = rep.get('scores', {})
            win = rep.get('object_name')
            ok = (win == m.name)
            n_ok += ok
            n_total += 1
            print(f"{m.name[:21]:<22}"
                  + "".join(f"{scores.get(n, 0.0):>{width}.3f}" for n in names)
                  + f"{(win or '-')[:17]:>19}{'  OK' if ok else 'MISS':>5}"
                  + f"{rep.get('margin', 0.0):>8.3f}")
            self_score = scores.get(m.name, 0.0)
            if self_score < 0.25:
                weak_diagonal.append((m.name, v, self_score))
            if not ok:
                confusions.append((m.name, win, rep.get('margin', 0.0)))

    print(f"\n{n_ok}/{n_total} correct ({100.0*n_ok/max(n_total,1):.0f}%)")

    if weak_diagonal:
        print("\nModels that scored poorly against THEMSELVES -- these need a finer "
              "per-model sampling\nstep; they are not being confused with anything, "
              "they are simply not being found:")
        for name, v, s in weak_diagonal:
            print(f"  {name:24s} view {v}: self-score {s:.3f}")
        worst = sorted({w[0] for w in weak_diagonal})
        print("\n  python scripts/build_ppf_library.py --sampling-step 0.025 \\")
        for name in worst:
            rec = lib.get(name)
            print(f"      --add Data/Input/{rec.filename} \\")
        print(f"      --out {args.library}")

    if confusions:
        print("\nConfusions (truth -> winner, margin). A small margin between parts of "
              "similar size and\nshape is a property of your CAD set, not a defect; "
              "raise PPF_MIN_MARGIN so the server\nflags them instead of guessing:")
        for truth, win, margin in confusions:
            print(f"  {truth:24s} -> {str(win):24s} margin {margin:.3f}")

    return 0 if n_ok == n_total else 2


if __name__ == '__main__':
    raise SystemExit(main())
