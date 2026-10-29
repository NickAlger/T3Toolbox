"""Read-only plan check (v6, M6): the SPECIFIED canonical zero-frame recipe, implemented exactly as the plan text says:
  per sharing group g (singletons are their own group), key k_g = the group's first mode index;
  rs = np.random.RandomState([ZERO_FRAME_SEED, k_g])                       (a per-group stream: no cross-mode coupling)
  G  = rs.standard_normal((n_g, N_g))                                      (row-major: the first m rows are the stream
                                                                            prefix, the same for every n_g >= m)
  Q, R = qr(G.T); Q *= sign(diag R)  ->  U_g = Q.T  (n_g x N_g)            (column j of Q depends only on rows <= j of G)
  TT cores identity-prefix; L[d-1] = R[0] = 0.
Checks: (1) a mode's factor is unchanged when ANOTHER mode's rank (or size) changes; (2) NESTED across a continuation
ladder: level r's factor is the leading-row prefix of level r+1's (to fp); (3) the zero reference over ladders is
non-decreasing (several data/seeds); (4) non-collapse on trailing-supported data; (5) layer independence + the O2 pairing
(44 pairs) under THIS recipe."""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, "tests/oracle_sweeps")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.optimizers as topt, t3toolbox.corewise as cw
src = open(os.path.join(HERE, 'c36_explicit_zero_frame.py')).read(); ns = {'__file__': os.path.join(HERE, 'c36_explicit_zero_frame.py')}
exec(compile(src[:src.index('rng = np.random.default_rng(0)')], 'c36_defs', 'exec'), ns)
zero_frame_like = ns['zero_frame_like']
ZERO_FRAME_SEED = 0

def haar_rows(n, N, key, seed=ZERO_FRAME_SEED):
    G = np.random.RandomState([seed, key]).standard_normal((n, N))
    Q, R = np.linalg.qr(G.T)
    return (Q * np.where(np.diag(R) >= 0, 1.0, -1.0)).T

def spec_zero_frame(frame, sharing=None):
    U, O, L, R = zero_frame_like(frame)
    d = len(U)
    labels = list(range(d)) if sharing is None else list(sharing)
    first = {}
    for i, lab in enumerate(labels):
        first.setdefault(lab, i)
    return (tuple(haar_rows(U[i].shape[0], U[i].shape[1], first[labels[i]]) for i in range(d)), O, L, R)

def frame_of(shape, tk, tt):
    geom = bgeo.ManifoldGeometryOps()
    return geom.frame(t3.TuckerTensorTrain.zeros(shape, tk, tt).data)

def ref_at(geom, kind, sample, data, F):
    g = geom.project(F, kind.transpose(-data if not isinstance(data, list) else [-z for z in data], sample, F,
                                       kind.precompute(F, sample)), aux=geom.precompute(F))
    return float(geom.inner(g, g)) ** 0.5

# (1) independence from other modes' ranks and sizes
F1 = spec_zero_frame(frame_of((8, 8, 8), (2, 3, 3), (1, 2, 3, 1)))
F2 = spec_zero_frame(frame_of((8, 8, 8), (3, 3, 3), (1, 3, 3, 1)))
F3 = spec_zero_frame(frame_of((11, 8, 8), (5, 3, 3), (1, 5, 3, 1)))
print('(1) mode-2 factor unchanged when mode 0 rank changes:', np.array_equal(F1[0][2], F2[0][2]),
      '| ... when mode 0 SIZE and rank change:', np.array_equal(F1[0][2], F3[0][2]))
# (2) nesting across a ladder, per mode
ladder = [((r, r, r), (1, r, r, 1)) for r in range(1, 8)]
frames = [spec_zero_frame(frame_of((8, 8, 8), tk, tt)) for tk, tt in ladder]
nest = max(float(np.abs(Fa[0][i] - Fb[0][i][:Fa[0][i].shape[0]]).max()) for Fa, Fb in zip(frames, frames[1:]) for i in range(3))
print(f'(2) nested across the ladder r = 1..7: max |U_i(r) - U_i(r+1)[:r]| = {nest:.1e}')
# (3) monotone reference over ladders, several problems
geom = bgeo.ManifoldGeometryOps()
mono_all = True
for seed in range(4):
    rng = np.random.default_rng(seed)
    shape, W = (8, 8, 8), 300
    ww = [rng.standard_normal((W, N)) for N in shape]
    data = np.asarray(t3.TuckerTensorTrain.randn(shape, (5, 5, 5), (1, 5, 5, 1)).apply(ww)) + 0.1 * rng.standard_normal(W)
    refs = [ref_at(geom, bfit.APPLY, ww, data, spec_zero_frame(frame_of(shape, tk, tt))) for tk, tt in ladder]
    mono = all(b >= a * (1 - 1e-12) for a, b in zip(refs, refs[1:]))
    mono_all &= mono
    print(f'(3) seed {seed}: refs over r = 1..7: ' + ' '.join('%.3e' % r for r in refs) + f'  non-decreasing {mono}')
# (4) non-collapse on trailing-supported data
rng = np.random.default_rng(1)
shape, tk, tt, W = (6, 7, 8), (2, 3, 2), (1, 2, 2, 1), 60
ww = [rng.standard_normal((W, N)) for N in shape]
ww = [np.concatenate([np.zeros((W, n)), w[:, n:]], axis=1) for w, n in zip(ww, tk)]
print(f'(4) trailing-supported data: reference {ref_at(geom, bfit.APPLY, ww, rng.standard_normal(W), spec_zero_frame(frame_of(shape, tk, tt))):.3e}')
# (5) O2 pairing under THIS recipe (the c37 harness with the spec frame)
import sweep_optimizers as SO, sweep_models as SM
def canonical_ref(geom_f, kind, sample, y, x0, order, weight, rep, sharing, shared):
    problem, init, _ = topt.least_squares_problem(geom_f, kind, sample, y, x0, order, weight, None, chunk_size=None) \
        if hasattr(topt, 'least_squares_problem') and callable(getattr(topt, 'least_squares_problem')) and False else \
        topt._setup(geom_f, kind, sample, y, x0, order, weight, None, chunk_size=None)
    lm = problem.local_model(cw.corewise_zeros_like(init))
    lab = sharing if shared else None
    if rep == 'ragged':
        F = spec_zero_frame(lm.frame, lab)
    else:
        Fr = spec_zero_frame(ufc.ut3frame_to_t3frame(lm.frame), lab)
        U, O, L, R = lm.frame[:4]
        F = ufc.t3frame_to_ut3frame(Fr, N=U.shape[-1], nU=U.shape[-2], nD=O.shape[-2], rL=L.shape[-1], rR=R.shape[-1])
    pk, geo = problem.kind, problem.geom
    resid = cw.corewise_scale(problem.data, -1.0) if isinstance(problem.data, (list, tuple)) else -problem.data
    g = geo.project(F, pk.transpose(resid, problem.sample, F, pk.precompute(F, problem.sample)), aux=geo.precompute(F))
    return float(geo.inner(g, g)) ** 0.5
worst_l = worst_r = 0.0; n = 0
for d in (3, 2):
    for shared in (False, True):
        for kind in ('apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives'):
            order = 2 if kind.endswith('_derivatives') else None
            wspecs = ['none'] + (['mode'] if kind == 'probe' else []) + (['order'] if kind.endswith('_derivatives') else []) \
                     + (['matrix'] if kind == 'probe_derivatives' else [])
            for wspec in wspecs:
                x, sharing, sample, weight, y = SO.setup(d, shared, kind, order, wspec, 11)
                refs, losses = {}, {}
                for rep in ('ragged', 'uniform'):
                    gf = SO.geometry_for(rep, 'manifold', shared, sharing)
                    x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
                    refs[rep] = canonical_ref(gf, kind, sample, y, x0, order, weight, rep, sharing, shared)
                    _, st = topt.newton_cg(gf, kind, sample, y, x0, order=order, weight=weight, max_newton=6, g0norm_newton=refs[rep])
                    losses[rep] = np.array(st['losses'])
                m = min(len(losses['ragged']), len(losses['uniform'])); n += 1
                worst_l = max(worst_l, SM.rel(losses['ragged'][:m], losses['uniform'][:m]))
                worst_r = max(worst_r, abs(refs['ragged'] - refs['uniform']) / refs['ragged'])
print(f'(5) {n} O2 pairs under the specified recipe: worst ref rel diff {worst_r:.1e}; worst paired newton_losses {worst_l:.1e} (threshold 1e-7)')
