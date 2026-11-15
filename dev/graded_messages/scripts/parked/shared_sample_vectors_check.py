"""PARKED IDEA (2026-11-14, low priority): take `xi` once per group when a sharing group's modes also receive the
same sample vector. Uses the prototype mockup beside this file, not the main mockup. See "Parked ideas" in EVALUATOR.md.

The unparked part of this check, kept for context:
How the evaluator meets the library's SHARING (the same Tucker factor at several modes).

In the library a shared T3 is an ordinary T3 whose group factors are equal, and sharing is handled by a geometry
wrapper (shared_geometry.py). The sampling kernels do not know about it. This script checks that the same holds
for the proposal: the per-sample part is untouched, and sharing enters only in core-sized steps around it.

  1. At a shared point the mockup's evaluator, unchanged, gives the unshared gradient and Gauss-Newton product.
  2. The shared gradient is the tied projection of that gradient; the shared Gauss-Newton product is the tied
     projection of the product along a tied direction. Both are core-sized steps after the sample loop.
  3. In the frame, the down Tucker cores D of a group are equal, so when a group's modes also receive the same
     sample vector, their `xi` components coincide. Otherwise no per-sample component coincides.

Usage:  python3 parked/shared_sample_vectors_check.py /path/to/T3Toolbox
"""
import importlib.util
import pathlib
import sys

import numpy as np

if len(sys.argv) > 1:
    sys.path.append(sys.argv[1])

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('plan_mockup', HERE / 'plan_mockup_per_group_xi.py')
pm = importlib.util.module_from_spec(_spec)
sys.modules['plan_mockup'] = pm
_spec.loader.exec_module(pm)


def main():
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.frame_variations_format as fvf
    import t3toolbox.manifold as t3m
    import t3toolbox.fitting as fitting
    import t3toolbox.shared_geometry as shared_geometry
    import t3toolbox.backend.probing as probing

    rng = np.random.default_rng(0)
    np.random.seed(0)
    sharing = (0, 0, 1, 0)                                   # modes 0, 1 and 3 use one Tucker factor
    shape, W = (9, 9, 7, 9), 200
    x = t3.TuckerTensorTrain.randn(shape, (4, 4, 3, 4), (1, 3, 4, 3, 1)).share(sharing)
    print('shared point: tucker ranks %s, tt ranks %s, factors tied: %s'
          % (x.tucker_ranks, x.tt_ranks, bool(np.all(x.has_shared_tucker_factors(sharing)))))
    w = [rng.standard_normal((W, N)) for N in shape]
    y = [rng.standard_normal((W, N)) for N in shape]
    r = [z - yy for z, yy in zip(probing.t3_probe(w, x.data), y)]

    geom = shared_geometry.shared_manifold(sharing)
    plain = fitting.probe_model(t3m.MANIFOLD, x, w, r)       # the unshared geometry at the same point
    tied = fitting.probe_model(geom, x, w, r)                # the shared geometry
    frame = plain.frame.data
    centers = pm.orthogonal_frame(x.data)[1]
    flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
    rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))
    coords = lambda tangent: tangent.data[1].data            # (V, H) of a T3Tangent
    as_tangent = lambda v: t3m.T3Tangent(plain.frame, fvf.T3Variations(*v))

    plan = pm.Plan(samples=64)
    p = coords(geom.randn(plain.frame))                      # a tied direction
    out = pm.evaluate(frame, centers, w, plan, want=('f', 'g'), y=y)
    Hp = pm.evaluate(frame, centers, w, plan, want=('Hp',), p=p)['Hp']

    print('1. the evaluator, unchanged, at the shared point, against the unshared geometry')
    print('   g %.0e   Hp %.0e' % (rel(out['g'], coords(plain.gradient)),
                                    rel(Hp, coords(plain.gn_hessian(as_tangent(p))))))
    print('2. sharing as a core-sized step after the sample loop, against the shared geometry')
    print('   tied projection of g  %.0e   tied projection of Hp %.0e'
          % (rel(coords(geom.project(as_tangent(out['g']))), coords(tied.gradient)),
             rel(coords(geom.project(as_tangent(Hp))), coords(tied.gn_hessian(as_tangent(p))))))

    group = [i for i, s in enumerate(sharing) if s == sharing[0]]
    w_same = [w[0] if i in group else w[i] for i in range(len(shape))]
    # 4. the same sample vector at the group's modes, declared
    r_same = [z - yy for z, yy in zip(probing.t3_probe(w_same, x.data), y)]
    tied_same = fitting.probe_model(geom, x, w_same, r_same)
    per_mode = pm.evaluate(frame, centers, w_same, plan, want=('f', 'g', 'xi'), y=y)
    grouped = pm.evaluate(frame, centers, w_same, plan, want=('f', 'g', 'xi'), y=y, sample_groups=sharing)
    Hp_mode = pm.evaluate(frame, centers, w_same, plan, want=('Hp',), p=p, have={'xi': per_mode['xi']})['Hp']
    Hp_group = pm.evaluate(frame, centers, w_same, plan, want=('Hp',), p=p, have={'xi': grouped['xi']},
                           sample_groups=sharing)['Hp']
    print('4. xi once per group (sample_groups=sharing), with the same sample vector at the group\'s modes')
    print('   against xi once per mode:        f %.0e   g %.0e   Hp %.0e'
          % (abs(grouped['f'] - per_mode['f']) / per_mode['f'], rel(grouped['g'], per_mode['g']), rel(Hp_group, Hp_mode)))
    print('   against the shared geometry:     tied projection of g %.0e   of Hp %.0e'
          % (rel(coords(geom.project(as_tangent(grouped['g']))), coords(tied_same.gradient)),
             rel(coords(geom.project(as_tangent(Hp_group))), coords(tied_same.gn_hessian(as_tangent(p))))))
    print('   xi arrays: %d per mode, %d per group;   bytes per sample: %d per mode, %d per group'
          % (len(per_mode['xi']), len(grouped['xi']), pm.tree_bytes(per_mode['xi']) // W, pm.tree_bytes(grouped['xi']) // W))



if __name__ == '__main__':
    main()
