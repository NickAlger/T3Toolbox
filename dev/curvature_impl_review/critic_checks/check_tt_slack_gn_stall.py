"""Critic check (read-only): F28/T25 claim 'on TT slack GN+Armijo does not converge in 80'. Is that GN's behaviour on the
PROBLEM, or an artefact of the slack frame? c30's TT-slack x0 is randn with tt ranks (1,9,9,1) but tucker (4,4,4): its
FRAME left ranks are [1,4,9,1], i.e. the TT bond (9) exceeds what the Tucker ranks can realize (4*4=16 >= 9 ok, but the
right chain is [1,4,4,1]). Compare GN from the slack point vs from the same tensor reduced to minimal ranks."""
import sys, math, io, contextlib; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.fv_conversions as fvc
for label, x_tk, x_tt in (('Tucker slack', (4, 4, 4), (1, 2, 2, 1)), ('TT slack', (4, 4, 4), (1, 9, 9, 1))):
    rng = np.random.default_rng(11); np.random.seed(11)
    shape, W = (6, 6, 6), 300
    target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(target.apply(ww))
    data = clean + 0.02 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, x_tk, x_tt)
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    minimal = t3.TuckerTensorTrain.get_minimal_ranks(x0.shape, x0.tucker_ranks, x0.tt_ranks)
    y, _, _ = x0.t3svd(); xm = y.rank_adjustment_sweep('right_to_left')
    for tag, xx in (('slack start', x0), ('same tensor, minimal ranks %s %s' % (tuple(minimal[0]), tuple(minimal[1])), xm)):
        g0 = None
        _, st = bopt.newton_cg(problem, xx.data, max_newton=80, gtol_rel=1e-8)
        h = st['history']
        e = [r['gnorm'] / h[0]['gnorm'] for r in h]
        tail = [b / a for a, b in zip(e[-6:], e[-5:])]
        print(f"{label:12s} {tag:42s}: GN newton {st['newton']:3d}, converged {h[-1]['converged']}, final e {e[-1]:.1e}, "
              f"last ratios {' '.join('%.2f' % t for t in tail)}, f {h[-1]['objective']:.6e}")
