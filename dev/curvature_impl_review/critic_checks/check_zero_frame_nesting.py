"""Critic check (read-only) for v5's canonical zero-point frame (C.2/C.15): does the per-level zero reference behave
consistently ACROSS RANK-CONTINUATION LEVELS? Two draw recipes for the seeded Haar Tucker factors:
  (A) v5's text: per group, a sign-fixed QR of an N_i x n_i draw from one sequential RandomState(0) stream;
  (B) c38's (verified) code: per group, an N_i x N_i draw, keep the first n_i columns (stream use independent of ranks).
(B) makes the frames NESTED across levels (QR column j depends only on draw columns <= j) and makes one mode's frame
independent of the other modes' ranks. Report the zero reference over a continuation rank ladder for each recipe."""
import sys, os; sys.path.insert(0, __import__("os").getcwd())
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "checks"))
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
src = open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'checks', 'c36_explicit_zero_frame.py')).read(); ns = {}
exec(compile(src[:src.index('rng = np.random.default_rng(0)')], 'c36_defs', 'exec'), ns)
zero_frame_like = ns['zero_frame_like']

def sign_fixed_q(M):
    Q, R = np.linalg.qr(M)
    return Q * np.where(np.diag(R) >= 0, 1.0, -1.0)

def haar_frame(frame, recipe, seed=0):
    U, O, L, R = zero_frame_like(frame)
    rs = np.random.RandomState(seed)
    Uh = []
    for u in U:
        n, N = u.shape
        if recipe == 'A':
            Uh.append(sign_fixed_q(rs.standard_normal((N, n)))[:, :n].T)
        else:
            Uh.append(sign_fixed_q(rs.standard_normal((N, N)))[:, :n].T)
    return (tuple(Uh), O, L, R)

def ref_at(geom, kind, sample, data, F):
    g = geom.project(F, kind.transpose(-data, sample, F, kind.precompute(F, sample)), aux=geom.precompute(F))
    return float(geom.inner(g, g)) ** 0.5

rng = np.random.default_rng(3); np.random.seed(3)
shape, W = (8, 8, 8), 400
ww = [rng.standard_normal((W, N)) for N in shape]
data = np.asarray(t3.TuckerTensorTrain.randn(shape, (5, 5, 5), (1, 5, 5, 1)).apply(ww))
geom = bgeo.ManifoldGeometryOps()
prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
ladder = [((1, 1, 1), (1, 1, 1, 1)), ((2, 2, 2), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1)), ((4, 4, 4), (1, 4, 4, 1)),
          ((5, 5, 5), (1, 5, 5, 1)), ((6, 6, 6), (1, 6, 6, 1))]
for recipe in ('A', 'B'):
    refs = []
    for tk, tt in ladder:
        z = t3.TuckerTensorTrain.zeros(shape, tk, tt)
        F = haar_frame(prob.local_model(z.data).frame, recipe)
        refs.append(ref_at(geom, bfit.APPLY, ww, data, F))
    mono = all(b >= a * (1 - 1e-12) for a, b in zip(refs, refs[1:]))
    print(f"recipe {recipe} ({'N x n draw, sequential stream (v5 text)' if recipe == 'A' else 'N x N draw, keep n (c38 code)'}): "
          + ' '.join(f'{r:.3e}' for r in refs) + f"  | non-decreasing over the ladder: {mono}")
# one mode's frame vs another mode's rank: change only mode 0's rank, compare mode 2's factor
z1 = t3.TuckerTensorTrain.zeros(shape, (2, 3, 3), (1, 2, 3, 1)); z2 = t3.TuckerTensorTrain.zeros(shape, (3, 3, 3), (1, 3, 3, 1))
for recipe in ('A', 'B'):
    F1 = haar_frame(prob.local_model(z1.data).frame, recipe); F2 = haar_frame(prob.local_model(z2.data).frame, recipe)
    print(f"recipe {recipe}: mode-2 factor unchanged when only mode 0's rank changes: {np.allclose(F1[0][2], F2[0][2])}")
