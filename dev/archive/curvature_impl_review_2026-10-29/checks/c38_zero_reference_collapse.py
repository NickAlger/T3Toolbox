"""Read-only plan check (v5, I-18a): any FIXED coordinate-aligned completion at X = 0 (today's LAPACK frame has identity-prefix
U; c33) gives a zero reference that COLLAPSES when the data gradient has no energy on the prefix coordinates -- e.g. apply
vectors supported on the trailing coordinates. A seeded Haar-random Tucker completion (host numpy, legacy RandomState(0),
sign-fixed QR) is generic. Also: the seeded-Haar reference is layer-independent (ragged vs packed uniform) and its spread
across seeds."""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf, t3toolbox.backend.ufv_conversions as ufc
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c36_explicit_zero_frame.py')).read(); ns = {}
exec(compile(src[:src.index('rng = np.random.default_rng(0)')], 'c36_defs', 'exec'), ns)
zero_frame_like = ns['zero_frame_like']

def haar_rows(n, N, rs):                    # n orthonormal rows, Haar-distributed, sign-fixed (unique given the draw)
    Q, R = np.linalg.qr(rs.standard_normal((N, N)))
    return (Q * np.sign(np.diag(R)))[:, :n].T

def haar_zero_frame_like(frame, groups=None, seed=0):
    U, O, L, R = zero_frame_like(frame)
    rs = np.random.RandomState(seed); drawn = {}
    Uh = []
    for i, u in enumerate(U):
        key = i if groups is None else groups[i]
        if key not in drawn:
            drawn[key] = haar_rows(u.shape[0], u.shape[1], rs)
        Uh.append(drawn[key])
    return (tuple(Uh), O, L, R)

def ref_at(geom, kind, sample, data, F):
    g = geom.project(F, kind.transpose(-data, sample, F, kind.precompute(F, sample)), aux=geom.precompute(F))
    return float(geom.inner(g, g)) ** 0.5

rng = np.random.default_rng(1); np.random.seed(1)
shape, tk, tt, W = (6, 7, 8), (2, 3, 2), (1, 2, 2, 1), 60
geom = bgeo.ManifoldGeometryOps()
zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
for label, aligned in (('generic Gaussian ww', False), ('ww supported on trailing coords', True)):
    ww = [rng.standard_normal((W, N)) for N in shape]
    if aligned:
        ww = [np.concatenate([np.zeros((W, n)), w[:, n:]], axis=1) for w, n in zip(ww, tk)]
    data = rng.standard_normal(W)
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
    lm = prob.local_model(zero.data)
    g = lm.gradient
    r_lapack = float(geom.inner(g, g)) ** 0.5
    r_prefix = ref_at(geom, bfit.APPLY, ww, data, zero_frame_like(lm.frame))
    r_haar = [ref_at(geom, bfit.APPLY, ww, data, haar_zero_frame_like(lm.frame, seed=s)) for s in range(6)]
    print(f"{label:34s}: LAPACK frame {r_lapack:.3e} | identity-prefix {r_prefix:.3e} | seeded Haar (seed 0) {r_haar[0]:.3e}, "
          f"seeds 0-5 spread {min(r_haar):.3e}-{max(r_haar):.3e}")
# layer independence of the seeded-Haar frame: pack the ragged construction into the uniform layout
ww = [rng.standard_normal((W, N)) for N in shape]; data = rng.standard_normal(W)
prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
F = haar_zero_frame_like(prob.local_model(zero.data).frame)
uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
ug = bgeo.UniformManifoldGeometryOps.from_point(uz.data)
uframe = ug.frame((uz.data[0], uz.data[1]))
UF = ufc.t3frame_to_ut3frame(F, N=uframe[0].shape[-1], nU=uframe[0].shape[-2], nD=uframe[1].shape[-2], rL=uframe[2].shape[-1], rR=uframe[3].shape[-1])
ukind = uf.uniform_sampling_kind('apply', uz.data, None); ps = uf.pack_sample('apply', ww, uz.N)
ra, ru = ref_at(geom, bfit.APPLY, ww, data, F), ref_at(ug, ukind, ps, data, UF)
print(f"seeded Haar, ragged {ra:.10e} vs packed uniform {ru:.10e} (rel {abs(ra - ru) / ra:.1e})")
