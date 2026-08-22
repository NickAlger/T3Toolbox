"""R5 edge cases: scalar-float residuals, entries index edge cases, error paths, d=1 oracles, compute_mu len."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as pr
import t3toolbox.backend.apply as ap
import t3toolbox.backend.entries as en
np.random.seed(0)
def hdr(s): print('\n=== ' + s)
def tryit(name, f):
    try:
        r = f(); print(f'{name}: OK ->', r if not hasattr(r, 'shape') else r.shape); return r
    except Exception as e:
        print(f'{name}: {type(e).__name__}: {e}'); return None

shape, tr, rr = (7, 5, 6), (3, 2, 4), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, rr); xd = x.to_dense()
frame, _ = bvf.t3_orthogonal_representations(x)
ww = [np.random.randn(N) for N in shape]

hdr('A. Python-float residual c (W=C=K=()): ambient accepts, tangent/corewise crash')
tryit('apply_ambient_transpose(1.7, ww)', lambda: t3.TuckerTensorTrain.apply_ambient_transpose(1.7, ww)[0])
tryit('T3Tangent.apply_transpose(1.7, ww, frame)', lambda: t3m.T3Tangent.apply_transpose(1.7, ww, frame))
tryit('T3Tangent.entries_transpose(1.7, (1,2,3), frame)', lambda: t3m.T3Tangent.entries_transpose(1.7, (1, 2, 3), frame))
tryit('x.apply_corewise_transpose(1.7, ww)', lambda: x.apply_corewise_transpose(1.7, ww)[0][0])
tryit('x.entries_corewise_transpose(1.7, (1,2,3))', lambda: x.entries_corewise_transpose(1.7, (1, 2, 3))[0][0])
tryit('np.float64 residual apply_transpose', lambda: t3m.T3Tangent.apply_transpose(np.float64(1.7), ww, frame))
tryit('0-d ndarray residual apply_transpose', lambda: t3m.T3Tangent.apply_transpose(np.asarray(1.7), ww, frame))

hdr('B. entries(): wrong-length LIST index -> docstring promises ValueError')
tryit('x.entries([1,2])', lambda: x.entries([1, 2]))
tryit('x.entries(np.array([1,2]))', lambda: x.entries(np.array([1, 2])))

hdr('C. entries with NEGATIVE indices: forward wraps (numpy), tangent/corewise wrap, ambient -> zero')
idx_neg = np.array([-1, -2, -3])
tryit('forward x.entries([-1,-2,-3])', lambda: float(x.entries(idx_neg)))
print('dense xd[-1,-2,-3] =', float(xd[-1, -2, -3]))
c = np.asarray(2.0)
fa = t3.TuckerTensorTrain.entries_ambient_transpose(c, idx_neg, shape)
T = t3.TuckerTensorTrain.from_canonical(fa).to_dense()
print('ambient: <from_canonical(E^T c), x>_F =', float(np.sum(T * xd)), ' expected c*entries =', 2.0 * float(xd[-1, -2, -3]),
      '; ||T|| =', float(np.linalg.norm(T)))
v = t3m.COREWISE.randn(frame)
ETc = t3m.T3Tangent.entries_transpose(c, idx_neg, frame, sum_over_probes=True)
print('tangent adjoint: <E^T c, v> =', float(sum(np.sum(a*b) for a, b in zip(ETc.variations.data[0]+ETc.variations.data[1], v.variations.data[0]+v.variations.data[1]))),
      ' c*v.entries =', 2.0 * float(v.entries(idx_neg)), ' dense:', 2.0 * float(v.to_dense()[-1, -2, -3]))
gU, gG = x.entries_corewise_transpose(c, idx_neg, sum_over_probes=True)
dU = [np.random.randn(*u.shape) for u in x.tucker_cores]; dG = [np.random.randn(*g.shape) for g in x.tt_cores]
def replace(kind, i, new):
    a, b = list(x.tucker_cores), list(x.tt_cores); (a if kind == 'U' else b)[i] = new
    return t3.TuckerTensorTrain(tuple(a), tuple(b))
Jd = sum(float(replace('U', i, dU[i]).entries(idx_neg)) for i in range(3)) + sum(float(replace('G', i, dG[i]).entries(idx_neg)) for i in range(3))
print('corewise adjoint: <g, d> =', float(sum(np.sum(a*b) for a, b in zip(gU+gG, dU+dG))), ' c*Jd =', 2.0 * Jd)

hdr('D. entries with OUT-OF-RANGE index: numpy raises; jax silently clamps')
idx_oor = np.array([7, 0, 0])   # N0 = 7 -> out of range
tryit('numpy forward x.entries([7,0,0])', lambda: float(x.entries(idx_oor)))
try:
    import jax, jax.numpy as jnp
    xj = x.to_jax()
    tryit('jax forward x.entries([7,0,0])', lambda: float(xj.entries(jnp.asarray(idx_oor))))
    print('   vs clamped dense xd[6,0,0] =', float(xd[6, 0, 0]))
    fj = t3.TuckerTensorTrain.entries_ambient_transpose(jnp.asarray(2.0), jnp.asarray(idx_oor), shape)
    print('   jax ambient ||T|| =', float(jnp.linalg.norm(t3.TuckerTensorTrain.from_canonical(fj).to_dense())))
    vj = t3m.COREWISE.randn(bvf.t3_orthogonal_representations(xj)[0])
    tryit('jax tangent entries [7,0,0]', lambda: float(vj.entries(jnp.asarray(idx_oor))))
except ImportError:
    print('jax not available')

hdr('E. duplicate indices with sum_over_probes=True scatter-add (adjoint identity)')
idx_dup = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])   # (d,)+W, W=(3,), all identical
cW = np.random.randn(3)
ETc = t3m.T3Tangent.entries_transpose(cW, idx_dup, frame, sum_over_probes=True)
lhs = float(np.sum(cW * v.entries(idx_dup)))
rhs = float(sum(np.sum(a*b) for a, b in zip(ETc.variations.data[0]+ETc.variations.data[1], v.variations.data[0]+v.variations.data[1])))
print('tangent dup adjoint:', lhs, rhs)
fa = t3.TuckerTensorTrain.entries_ambient_transpose(cW, idx_dup, shape, sum_over_probes=True)
T = t3.TuckerTensorTrain.from_canonical(fa).to_dense()
print('ambient dup: T[1,2,3] =', float(T[1, 2, 3]), ' sum(c) =', float(cW.sum()), ' ||T||=', float(np.linalg.norm(T)))

hdr('F. apply with W=() vs W=(1,) (single vector)')
a0 = x.apply(ww); a1 = x.apply([w[None] for w in ww])
print('W=():', np.shape(a0), float(a0), ' W=(1,):', np.shape(a1), float(a1[0]))
z0 = x.probe(ww); z1 = x.probe([w[None] for w in ww]); print('probe W=(1,) shapes', [z.shape for z in z1], np.allclose(z1[0][0], z0[0]))

hdr('G. d=1 oracles: dense_probe with W; project_ambient(t3) at d=1')
x1 = t3.TuckerTensorTrain.randn((7,), (3,), (1, 1)); w1 = [np.random.randn(2, 7)]
tryit('x1.probe(W=(2,))', lambda: x1.probe(w1)[0])
tryit('dense_probe(W=(2,), d=1)', lambda: pr.dense_probe(w1, x1.to_dense())[0])
tryit('dense_probe(W=(), d=1)', lambda: pr.dense_probe([w1[0][0]], x1.to_dense())[0])
fr1, _ = bvf.t3_orthogonal_representations(x1)
g1 = t3.TuckerTensorTrain.randn((7,), (3,), (1, 1))
tryit('MANIFOLD.project_ambient(frame_d1, t3)', lambda: t3m.MANIFOLD.project_ambient(fr1, g1))
tryit('MANIFOLD.project_ambient(frame_d1, dense)', lambda: t3m.MANIFOLD.project_ambient(fr1, g1.to_dense()))
tryit('MANIFOLD.transport at d=1', lambda: t3m.MANIFOLD.transport(t3m.MANIFOLD.randn(fr1), fr1))
tryit('MANIFOLD.project(tangent) at d=1', lambda: t3m.MANIFOLD.project(t3m.COREWISE.randn(fr1)))
x1b = t3.TuckerTensorTrain.randn((7, 5), (3, 2), (1, 2, 1)); fr2, _ = bvf.t3_orthogonal_representations(x1b)
tryit('MANIFOLD.project_ambient(frame_d2, t3)', lambda: t3m.MANIFOLD.project_ambient(fr2, t3.TuckerTensorTrain.randn((7, 5), (3, 2), (1, 2, 1))))

hdr('H. compute_mu shape comment says left_tt_cores len=d-1; t3_probe passes len=d. Passing d-1 silently truncates:')
xis = pr.compute_xi(x.tucker_cores, ww)
mus_d = pr.compute_mu(x.tt_cores, xis); mus_dm1 = pr.compute_mu(x.tt_cores[:-1], xis)
print('len(mus) with d cores:', len(mus_d), ' with d-1 cores:', len(mus_dm1))
etas = pr.compute_eta(x.tt_cores, mus_dm1, pr.compute_nu(x.tt_cores, xis))
print('compute_eta with the d-1 mus -> len(etas) =', len(etas), '(silently d-1; no error)')

hdr('I. jit smoke of the from_sweep hooks (ragged, jax)')
try:
    import jax, jax.numpy as jnp
    xj = x.to_jax(); frj, _ = bvf.t3_orthogonal_representations(xj); vj = t3m.COREWISE.randn(frj, stack_shape=(2,))
    wwj = [jnp.asarray(np.random.randn(3, N)) for N in shape]; idxj = jnp.asarray(np.array([np.random.randint(0, N, size=(3,)) for N in shape]))
    cj = jnp.asarray(np.random.randn(3, 2))
    f = jax.jit(lambda vr, fr: ap.tv_apply_transpose_from_sweep(cj, wwj, fr, ap.tv_precompute_apply_frame_sweep(fr, wwj), True))
    r = f(vj.variations.data, frj.data); print('jit apply_T_from_sweep:', [a.shape for a in r[0]])
    f = jax.jit(lambda vr, fr: en.tv_entries_jacobian_from_sweep(vr, idxj, fr, en.tv_precompute_entries_frame_sweep(fr, idxj)))
    r = f(vj.variations.data, frj.data); print('jit entries_jac_from_sweep:', r.shape)
    zj = [jnp.asarray(np.random.randn(3, 2, N)) for N in shape]
    f = jax.jit(lambda vr, fr: pr.tv_probe_transpose_from_sweep(zj, wwj, fr, pr.tv_precompute_probe_frame_sweep(fr, wwj), False))
    r = f(vj.variations.data, frj.data); print('jit probe_T_from_sweep:', [a.shape for a in r[1]])
except ImportError:
    print('jax not available')
